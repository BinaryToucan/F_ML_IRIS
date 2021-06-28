FROM mcr.microsoft.com/dotnet/core/runtime:3.1 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1 AS build
WORKDIR /src
COPY ["F_ML_CORE_2_MEAN/F_ML_CORE_2_MEAN.fsproj", "F_ML_CORE_2_MEAN/"]
RUN dotnet restore "F_ML_CORE_2_MEAN/F_ML_CORE_2_MEAN.fsproj"
COPY . .
WORKDIR "/src/F_ML_CORE_2_MEAN"
RUN dotnet build "F_ML_CORE_2_MEAN.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "F_ML_CORE_2_MEAN.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "F_ML_CORE_2_MEAN.dll"]
