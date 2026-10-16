using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AutoPlayQA
{
    internal sealed class AutoPlayQARequest
    {
        public string Id;
        public string Type;
        public string Protocol;
        public int Seed;
        public bool HasSeed;
        public string Control;
    }

    /// <summary>
    /// Small protocol-specific JSON codec. It deliberately supports only the
    /// request envelope and response DTOs used by the bridge; it is not a
    /// general-purpose JSON library.
    /// </summary>
    internal static class AutoPlayQAProtocolJson
    {
        public static bool TryParseRequest(
            string json,
            out AutoPlayQARequest request,
            out string errorCode,
            out string errorMessage)
        {
            request = null;
            errorCode = null;
            errorMessage = null;

            var parsed = new AutoPlayQARequest();
            request = parsed;
            var reader = new RequestReader(json);
            if (!reader.TryRead(parsed, out errorCode, out errorMessage))
            {
                return false;
            }

            if (string.IsNullOrEmpty(parsed.Id))
            {
                errorCode = "invalid_request";
                errorMessage = "Request field 'id' must be a non-empty string.";
                return false;
            }

            if (parsed.Id.Length > 256)
            {
                errorCode = "invalid_request";
                errorMessage = "Request field 'id' is too long.";
                return false;
            }

            if (string.IsNullOrEmpty(parsed.Type))
            {
                errorCode = "invalid_request";
                errorMessage = "Request field 'type' must be a non-empty string.";
                return false;
            }

            return true;
        }

        public static string Hello(
            string id,
            AutoPlayQAGameInfo game,
            string[] controls,
            AutoPlayQACapabilities capabilities)
        {
            var builder = BeginSuccess(id, "hello");
            builder.Append(",\"protocol\":");
            AppendString(builder, AutoPlayQAProtocol.Version);
            builder.Append(",\"game\":{");
            builder.Append("\"id\":");
            AppendString(builder, game == null ? null : game.Id);
            builder.Append(",\"version\":");
            AppendString(builder, game == null ? null : game.Version);
            builder.Append('}');
            builder.Append(",\"controls\":");
            AppendStringArray(builder, controls);
            builder.Append(",\"capabilities\":{");
            AppendBooleanProperty(builder, "grid", capabilities != null && capabilities.Grid, false);
            AppendBooleanProperty(builder, "legend", capabilities != null && capabilities.Legend, true);
            AppendBooleanProperty(builder, "entities", capabilities != null && capabilities.Entities, true);
            AppendBooleanProperty(builder, "reset", capabilities != null && capabilities.Reset, true);
            AppendBooleanProperty(builder, "seed", capabilities != null && capabilities.Seed, true);
            AppendBooleanProperty(
                builder,
                "defect_injection",
                capabilities != null && capabilities.DefectInjection,
                true);

            // These are additive autoplay-qa/1 extensions. Omit false values
            // so legacy adapters preserve their original handshake exactly.
            if (capabilities != null && capabilities.Continuous)
            {
                AppendBooleanProperty(builder, "continuous", true, true);
            }

            if (capabilities != null && capabilities.SolidsVisible)
            {
                AppendBooleanProperty(builder, "solids_visible", true, true);
            }

            builder.Append("}}");
            return builder.ToString();
        }

        public static string Observation(
            string id,
            string type,
            AutoPlayQAObservation observation,
            AutoPlayQACapabilities capabilities)
        {
            var builder = BeginSuccess(id, type);
            builder.Append(",\"observation\":");
            AppendObservation(builder, observation, capabilities);
            builder.Append('}');
            return builder.ToString();
        }

        public static string Close(string id)
        {
            var builder = BeginSuccess(id, "close");
            builder.Append('}');
            return builder.ToString();
        }

        public static string Error(string id, string code, string message, bool retryable)
        {
            var builder = new StringBuilder(192);
            builder.Append("{\"id\":");
            AppendString(builder, id);
            builder.Append(",\"type\":\"error\",\"ok\":false,\"error\":{");
            builder.Append("\"code\":");
            AppendString(builder, code);
            builder.Append(",\"message\":");
            AppendString(builder, message);
            builder.Append(",\"retryable\":");
            builder.Append(retryable ? "true" : "false");
            builder.Append("}}");
            return builder.ToString();
        }

        private static StringBuilder BeginSuccess(string id, string type)
        {
            var builder = new StringBuilder(512);
            builder.Append("{\"id\":");
            AppendString(builder, id);
            builder.Append(",\"type\":");
            AppendString(builder, type);
            builder.Append(",\"ok\":true");
            return builder;
        }

        private static void AppendObservation(
            StringBuilder builder,
            AutoPlayQAObservation observation,
            AutoPlayQACapabilities capabilities)
        {
            if (observation == null)
            {
                builder.Append("null");
                return;
            }

            builder.Append('{');
            builder.Append("\"tick\":");
            builder.Append(observation.Tick.ToString(CultureInfo.InvariantCulture));
            builder.Append(",\"status\":");
            AppendString(builder, observation.Status);
            builder.Append(",\"player\":");
            AppendPlayer(builder, observation.Player);
            builder.Append(",\"entities\":");
            AppendEntities(builder, observation.Entities);
            builder.Append(",\"events\":");
            AppendStringArray(builder, observation.Events);
            builder.Append(",\"controls\":");
            AppendStringArray(builder, observation.Controls);

            if (capabilities != null && capabilities.Grid && observation.Grid != null)
            {
                builder.Append(",\"grid\":");
                AppendStringArray(builder, observation.Grid);
            }

            if (capabilities != null && capabilities.Legend && observation.Legend != null)
            {
                builder.Append(",\"legend\":");
                AppendLegend(builder, observation.Legend);
            }

            builder.Append('}');
        }

        private static void AppendPlayer(StringBuilder builder, AutoPlayQAPlayer player)
        {
            builder.Append('{');
            if (player != null && player.Position != null)
            {
                builder.Append("\"position\":");
                AppendNumberArray(builder, player.Position);
                builder.Append(',');
            }

            if (player != null && player.Velocity != null)
            {
                builder.Append("\"velocity\":");
                AppendNumberArray(builder, player.Velocity);
                builder.Append(',');
            }

            if (player != null && player.Grounded.HasValue)
            {
                builder.Append("\"grounded\":");
                builder.Append(player.Grounded.Value ? "true" : "false");
                builder.Append(',');
            }

            builder.Append("\"inventory\":");
            AppendStringArray(builder, player == null ? null : player.Inventory);
            builder.Append('}');
        }

        private static void AppendEntities(StringBuilder builder, AutoPlayQAEntity[] entities)
        {
            builder.Append('[');
            if (entities != null)
            {
                for (var index = 0; index < entities.Length; index++)
                {
                    if (index > 0)
                    {
                        builder.Append(',');
                    }

                    var entity = entities[index];
                    builder.Append('{');
                    builder.Append("\"id\":");
                    AppendString(builder, entity == null ? null : entity.Id);
                    builder.Append(",\"kind\":");
                    AppendString(builder, entity == null ? null : entity.Kind);
                    if (entity != null && entity.Position != null)
                    {
                        builder.Append(",\"position\":");
                        AppendNumberArray(builder, entity.Position);
                    }

                    if (entity != null && entity.Size != null)
                    {
                        builder.Append(",\"size\":");
                        AppendNumberArray(builder, entity.Size);
                    }

                    builder.Append(",\"state\":");
                    AppendState(builder, entity == null ? null : entity.State);
                    builder.Append('}');
                }
            }

            builder.Append(']');
        }

        private static void AppendState(StringBuilder builder, AutoPlayQAStateEntry[] entries)
        {
            builder.Append('{');
            var wroteProperty = false;
            if (entries != null)
            {
                for (var index = 0; index < entries.Length; index++)
                {
                    var entry = entries[index];
                    if (entry == null || string.IsNullOrEmpty(entry.Key))
                    {
                        continue;
                    }

                    if (wroteProperty)
                    {
                        builder.Append(',');
                    }

                    AppendString(builder, entry.Key);
                    builder.Append(':');
                    switch (entry.Kind)
                    {
                        case AutoPlayQAValueKind.String:
                            AppendString(builder, entry.StringValue);
                            break;
                        case AutoPlayQAValueKind.Boolean:
                            builder.Append(entry.BooleanValue ? "true" : "false");
                            break;
                        case AutoPlayQAValueKind.Number:
                            if (double.IsNaN(entry.NumberValue) || double.IsInfinity(entry.NumberValue))
                            {
                                builder.Append("null");
                            }
                            else
                            {
                                builder.Append(entry.NumberValue.ToString("R", CultureInfo.InvariantCulture));
                            }

                            break;
                        default:
                            builder.Append("null");
                            break;
                    }

                    wroteProperty = true;
                }
            }

            builder.Append('}');
        }

        private static void AppendLegend(StringBuilder builder, AutoPlayQALegendEntry[] entries)
        {
            builder.Append('{');
            var wroteProperty = false;
            for (var index = 0; index < entries.Length; index++)
            {
                var entry = entries[index];
                if (entry == null || string.IsNullOrEmpty(entry.Glyph))
                {
                    continue;
                }

                if (wroteProperty)
                {
                    builder.Append(',');
                }

                AppendString(builder, entry.Glyph);
                builder.Append(':');
                AppendString(builder, entry.Meaning);
                wroteProperty = true;
            }

            builder.Append('}');
        }

        private static void AppendStringArray(StringBuilder builder, string[] values)
        {
            builder.Append('[');
            if (values != null)
            {
                for (var index = 0; index < values.Length; index++)
                {
                    if (index > 0)
                    {
                        builder.Append(',');
                    }

                    AppendString(builder, values[index]);
                }
            }

            builder.Append(']');
        }

        private static void AppendNumberArray(StringBuilder builder, double[] values)
        {
            builder.Append('[');
            for (var index = 0; index < values.Length; index++)
            {
                if (index > 0)
                {
                    builder.Append(',');
                }

                var value = values[index];
                if (double.IsNaN(value) || double.IsInfinity(value))
                {
                    builder.Append("null");
                }
                else
                {
                    builder.Append(value.ToString("R", CultureInfo.InvariantCulture));
                }
            }

            builder.Append(']');
        }

        private static void AppendBooleanProperty(
            StringBuilder builder,
            string name,
            bool value,
            bool prependComma)
        {
            if (prependComma)
            {
                builder.Append(',');
            }

            AppendString(builder, name);
            builder.Append(':');
            builder.Append(value ? "true" : "false");
        }

        private static void AppendString(StringBuilder builder, string value)
        {
            if (value == null)
            {
                builder.Append("null");
                return;
            }

            builder.Append('"');
            for (var index = 0; index < value.Length; index++)
            {
                var character = value[index];
                switch (character)
                {
                    case '"':
                        builder.Append("\\\"");
                        break;
                    case '\\':
                        builder.Append("\\\\");
                        break;
                    case '\b':
                        builder.Append("\\b");
                        break;
                    case '\f':
                        builder.Append("\\f");
                        break;
                    case '\n':
                        builder.Append("\\n");
                        break;
                    case '\r':
                        builder.Append("\\r");
                        break;
                    case '\t':
                        builder.Append("\\t");
                        break;
                    default:
                        if (character < 0x20)
                        {
                            builder.Append("\\u");
                            builder.Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            builder.Append(character);
                        }

                        break;
                }
            }

            builder.Append('"');
        }

        private sealed class RequestReader
        {
            private readonly string _json;
            private readonly HashSet<string> _fields = new HashSet<string>(StringComparer.Ordinal);
            private int _position;

            public RequestReader(string json)
            {
                _json = json;
            }

            public bool TryRead(
                AutoPlayQARequest request,
                out string errorCode,
                out string error)
            {
                errorCode = "invalid_json";
                error = "Request must be one JSON object.";
                if (_json == null)
                {
                    return false;
                }

                SkipWhitespace();
                if (!AtEnd && Peek != '{')
                {
                    if (TrySkipValue(0))
                    {
                        SkipWhitespace();
                        if (AtEnd)
                        {
                            errorCode = "invalid_message";
                            error = "Protocol messages must be JSON objects.";
                        }
                    }

                    return false;
                }

                if (!Take('{'))
                {
                    return false;
                }

                SkipWhitespace();
                if (Take('}'))
                {
                    SkipWhitespace();
                    return AtEnd;
                }

                while (true)
                {
                    string key;
                    if (!TryReadString(out key))
                    {
                        error = "JSON object keys must be strings.";
                        return false;
                    }

                    if (!_fields.Add(key))
                    {
                        errorCode = "invalid_request";
                        error = "Duplicate request fields are not allowed.";
                        return false;
                    }

                    SkipWhitespace();
                    if (!Take(':'))
                    {
                        error = "A request field is missing ':'.";
                        return false;
                    }

                    SkipWhitespace();
                    if (!TryReadField(key, request, out errorCode, out error))
                    {
                        return false;
                    }

                    SkipWhitespace();
                    if (Take('}'))
                    {
                        SkipWhitespace();
                        if (!AtEnd)
                        {
                            error = "Unexpected data follows the request object.";
                            return false;
                        }

                        return true;
                    }

                    if (!Take(','))
                    {
                        error = "Request object fields must be separated by ','.";
                        return false;
                    }

                    SkipWhitespace();
                }
            }

            private bool TryReadField(
                string key,
                AutoPlayQARequest request,
                out string errorCode,
                out string error)
            {
                errorCode = "invalid_json";
                error = null;
                string value;
                switch (key)
                {
                    case "id":
                        if (!TryReadString(out value))
                        {
                            errorCode = "invalid_request";
                            error = "Request field 'id' must be a string.";
                            return false;
                        }

                        request.Id = value;
                        return true;
                    case "type":
                        if (!TryReadString(out value))
                        {
                            errorCode = "invalid_request";
                            error = "Request field 'type' must be a string.";
                            return false;
                        }

                        request.Type = value;
                        return true;
                    case "protocol":
                        if (!TryReadString(out value))
                        {
                            errorCode = "unsupported_protocol";
                            error = "Request field 'protocol' must be a string.";
                            return false;
                        }

                        request.Protocol = value;
                        return true;
                    case "control":
                        if (!TryReadString(out value))
                        {
                            errorCode = "invalid_control";
                            error = "Request field 'control' must be a string.";
                            return false;
                        }

                        request.Control = value;
                        return true;
                    case "seed":
                        int seed;
                        if (!TryReadInteger(out seed))
                        {
                            errorCode = "invalid_seed";
                            error = "Request field 'seed' must be a 32-bit integer.";
                            return false;
                        }

                        request.Seed = seed;
                        request.HasSeed = true;
                        return true;
                    default:
                        if (!TrySkipValue(0))
                        {
                            error = "An unknown request field has an invalid JSON value.";
                            return false;
                        }

                        return true;
                }
            }

            private bool TryReadInteger(out int value)
            {
                value = 0;
                var start = _position;
                if (!TryReadNumber())
                {
                    return false;
                }

                var token = _json.Substring(start, _position - start);
                if (token.IndexOf('.') >= 0 || token.IndexOf('e') >= 0 || token.IndexOf('E') >= 0)
                {
                    return false;
                }

                return int.TryParse(
                    token,
                    NumberStyles.AllowLeadingSign,
                    CultureInfo.InvariantCulture,
                    out value);
            }

            private bool TrySkipValue(int depth)
            {
                if (depth > 64 || AtEnd)
                {
                    return false;
                }

                if (Peek == '"')
                {
                    string ignored;
                    return TryReadString(out ignored);
                }

                if (Peek == '{')
                {
                    _position++;
                    SkipWhitespace();
                    if (Take('}'))
                    {
                        return true;
                    }

                    while (true)
                    {
                        string ignored;
                        if (!TryReadString(out ignored))
                        {
                            return false;
                        }

                        SkipWhitespace();
                        if (!Take(':'))
                        {
                            return false;
                        }

                        SkipWhitespace();
                        if (!TrySkipValue(depth + 1))
                        {
                            return false;
                        }

                        SkipWhitespace();
                        if (Take('}'))
                        {
                            return true;
                        }

                        if (!Take(','))
                        {
                            return false;
                        }

                        SkipWhitespace();
                    }
                }

                if (Peek == '[')
                {
                    _position++;
                    SkipWhitespace();
                    if (Take(']'))
                    {
                        return true;
                    }

                    while (true)
                    {
                        if (!TrySkipValue(depth + 1))
                        {
                            return false;
                        }

                        SkipWhitespace();
                        if (Take(']'))
                        {
                            return true;
                        }

                        if (!Take(','))
                        {
                            return false;
                        }

                        SkipWhitespace();
                    }
                }

                if (TryTakeLiteral("true") || TryTakeLiteral("false") || TryTakeLiteral("null"))
                {
                    return true;
                }

                return TryReadFiniteNumber();
            }

            private bool TryReadFiniteNumber()
            {
                var start = _position;
                if (!TryReadNumber())
                {
                    return false;
                }

                double value;
                var token = _json.Substring(start, _position - start);
                if (!double.TryParse(
                    token,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out value) || double.IsNaN(value) || double.IsInfinity(value))
                {
                    _position = start;
                    return false;
                }

                return true;
            }

            private bool TryReadNumber()
            {
                var start = _position;
                Take('-');
                if (Take('0'))
                {
                    if (!AtEnd && char.IsDigit(Peek))
                    {
                        _position = start;
                        return false;
                    }
                }
                else
                {
                    if (AtEnd || Peek < '1' || Peek > '9')
                    {
                        _position = start;
                        return false;
                    }

                    while (!AtEnd && char.IsDigit(Peek))
                    {
                        _position++;
                    }
                }

                if (Take('.'))
                {
                    if (AtEnd || !char.IsDigit(Peek))
                    {
                        _position = start;
                        return false;
                    }

                    while (!AtEnd && char.IsDigit(Peek))
                    {
                        _position++;
                    }
                }

                if (!AtEnd && (Peek == 'e' || Peek == 'E'))
                {
                    _position++;
                    if (!AtEnd && (Peek == '+' || Peek == '-'))
                    {
                        _position++;
                    }

                    if (AtEnd || !char.IsDigit(Peek))
                    {
                        _position = start;
                        return false;
                    }

                    while (!AtEnd && char.IsDigit(Peek))
                    {
                        _position++;
                    }
                }

                return _position > start;
            }

            private bool TryReadString(out string value)
            {
                value = null;
                if (!Take('"'))
                {
                    return false;
                }

                var builder = new StringBuilder();
                while (!AtEnd)
                {
                    var character = _json[_position++];
                    if (character == '"')
                    {
                        value = builder.ToString();
                        return true;
                    }

                    if (character < 0x20)
                    {
                        return false;
                    }

                    if (character != '\\')
                    {
                        builder.Append(character);
                        continue;
                    }

                    if (AtEnd)
                    {
                        return false;
                    }

                    character = _json[_position++];
                    switch (character)
                    {
                        case '"':
                        case '\\':
                        case '/':
                            builder.Append(character);
                            break;
                        case 'b':
                            builder.Append('\b');
                            break;
                        case 'f':
                            builder.Append('\f');
                            break;
                        case 'n':
                            builder.Append('\n');
                            break;
                        case 'r':
                            builder.Append('\r');
                            break;
                        case 't':
                            builder.Append('\t');
                            break;
                        case 'u':
                            int codePoint;
                            if (!TryReadHex4(out codePoint))
                            {
                                return false;
                            }

                            builder.Append((char)codePoint);
                            break;
                        default:
                            return false;
                    }
                }

                return false;
            }

            private bool TryReadHex4(out int value)
            {
                value = 0;
                if (_position + 4 > _json.Length)
                {
                    return false;
                }

                for (var index = 0; index < 4; index++)
                {
                    var character = _json[_position++];
                    int digit;
                    if (character >= '0' && character <= '9')
                    {
                        digit = character - '0';
                    }
                    else if (character >= 'a' && character <= 'f')
                    {
                        digit = character - 'a' + 10;
                    }
                    else if (character >= 'A' && character <= 'F')
                    {
                        digit = character - 'A' + 10;
                    }
                    else
                    {
                        return false;
                    }

                    value = (value * 16) + digit;
                }

                return true;
            }

            private bool TryTakeLiteral(string literal)
            {
                if (_position + literal.Length > _json.Length)
                {
                    return false;
                }

                for (var index = 0; index < literal.Length; index++)
                {
                    if (_json[_position + index] != literal[index])
                    {
                        return false;
                    }
                }

                _position += literal.Length;
                return true;
            }

            private void SkipWhitespace()
            {
                while (!AtEnd)
                {
                    var character = Peek;
                    if (character != ' ' && character != '\t' && character != '\r' && character != '\n')
                    {
                        return;
                    }

                    _position++;
                }
            }

            private bool Take(char expected)
            {
                if (AtEnd || Peek != expected)
                {
                    return false;
                }

                _position++;
                return true;
            }

            private char Peek
            {
                get { return _json[_position]; }
            }

            private bool AtEnd
            {
                get { return _position >= _json.Length; }
            }
        }
    }
}
