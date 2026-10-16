CREATE TABLE `jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`request_id` text NOT NULL,
	`game` text NOT NULL,
	`profile` text NOT NULL,
	`seed` integer NOT NULL,
	`max_usd` real NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	`status` text NOT NULL,
	`active_slot` integer,
	`runner_id` text,
	`lease_id` text,
	`cancel_requested` integer DEFAULT 0 NOT NULL,
	`revision` integer DEFAULT 0 NOT NULL,
	`summary` text NOT NULL,
	`snapshot` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `jobs_request_id_unique` ON `jobs` (`request_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `one_active_job` ON `jobs` (`active_slot`);--> statement-breakpoint
CREATE TABLE `runner_presence` (
	`slot` integer PRIMARY KEY NOT NULL,
	`runner_id` text NOT NULL,
	`last_seen` text NOT NULL,
	`unity_ready` integer DEFAULT 0 NOT NULL
);
