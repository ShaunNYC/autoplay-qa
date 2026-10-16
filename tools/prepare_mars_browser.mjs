/** Copy and build the original Mars UI locally. Never modify original source. */
import { cp, mkdir, readFile, readdir, lstat, readlink, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const INPUTS = ['src', 'public', 'index.html', 'package.json', 'package-lock.json', 'vite.config.mjs'];
async function fileList(root, relative, allowLinks = false) {
  const candidate = path.join(root, relative); const stat = await lstat(candidate);
  if (stat.isSymbolicLink()) {
    if (!allowLinks) throw new Error('Source symlinks are not supported.');
    const target = await readlink(candidate);
    if (!path.resolve(path.dirname(candidate), target).startsWith(path.resolve(root) + path.sep)) throw new Error('Dependency symlink escapes the isolated copy.');
    return [{ path:relative, symlink:target }];
  }
  if (stat.isDirectory()) {
    const result = [];
    for (const name of (await readdir(candidate)).sort()) result.push(...await fileList(root, path.join(relative,name), allowLinks));
    return result;
  }
  return [{ path: relative, sha256: hash(await readFile(candidate)) }];
}
export async function prepareMarsBrowser(projectDir, outputDir) {
  const project = path.resolve(projectDir); const output = path.resolve(outputDir);
  if (output === project || output.startsWith(project + path.sep) || project.startsWith(output + path.sep)) throw new Error('The isolated copy must be outside the original project.');
  const before = (await Promise.all(INPUTS.map(name => fileList(project, name)))).flat();
  await mkdir(output, { recursive: false });
  for (const name of INPUTS) await cp(path.join(project,name), path.join(output,name), { recursive:true, errorOnExist:true, force:false });
  // Dependencies are copied too: no dev-server or bundler write can reach source.
  await cp(path.join(project,'node_modules'), path.join(output,'node_modules'), { recursive:true, verbatimSymlinks:true });
  const dependencyFiles = await fileList(output, 'node_modules', true);
  const require = createRequire(path.join(output,'package.json'));
  const { build } = await import(pathToFileURL(require.resolve('vite')).href);
  const react = (await import(pathToFileURL(require.resolve('@vitejs/plugin-react')).href)).default;
  const buildRecipe = { configFile:false, plugin:'@vitejs/plugin-react', outDir:'dist', sourcemap:false, emptyOutDir:true, sourceConfig:'vite.config.mjs frozen for provenance; isolated build uses this explicit equivalent application recipe' };
  await build({ root:output, configFile:false, logLevel:'silent', plugins:[react()], build:{ outDir:'dist', sourcemap:false, emptyOutDir:true } });
  const after = (await Promise.all(INPUTS.map(name => fileList(project,name)))).flat();
  if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Original source changed during isolation.');
  const files = await fileList(output, 'dist');
  const manifest = { format:'mars-browser-source-v1', sourceFiles:before, sourceSha256:hash(JSON.stringify(before)), buildFiles:files, buildSha256:hash(JSON.stringify(files)), sourceUnmodified:true, defectsInjected:false, dependencyPolicy:'copied-local-installed-dependencies', dependencyFileCount:dependencyFiles.length, dependencySha256:hash(JSON.stringify(dependencyFiles)), buildRecipe, buildRecipeSha256:hash(JSON.stringify(buildRecipe)), builderSha256:hash(await readFile(new URL(import.meta.url))), route:'/#/mars' };
  await writeFile(path.join(output,'source-manifest.json'), JSON.stringify(manifest,null,2)+'\n', { mode:0o600 });
  return manifest;
}
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const args = process.argv.slice(2); const value = name => args[args.indexOf(name)+1];
  try { process.stdout.write(JSON.stringify(await prepareMarsBrowser(value('--project'),value('--output')))+'\n'); }
  catch (error) { process.stderr.write('Isolated Mars preparation failed: '+error.message+'\n'); process.exitCode=1; }
}
