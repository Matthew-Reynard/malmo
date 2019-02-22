import numpy as np

missionXML='''<?xml version="1.0" encoding="UTF-8"?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Main mission</Summary>
  </About>

  <ModSettings>
    <MsPerTick>50</MsPerTick>
  </ModSettings>

  <ServerSection>
    <ServerInitialConditions>
        <Time>
          <StartTime>18000</StartTime>
          <AllowPassageOfTime>true</AllowPassageOfTime>
        </Time>
        <Weather>clear</Weather>
        <AllowSpawning>false</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,100*1,5*3,2;3;,biome_1"/>
      <!-- <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/> -->

      <!-- <DefaultWorldGenerator/> -->

      <!-- <ClassroomDecorator seed="0" palette="pyramid"> -->
      <!-- <ClassroomDecorator seed="random">
        <specification>
          <width>7</width>
          <height>7</height>
          <length>7</length>
          <pathLength>0</pathLength>
          <divisions>
            <southNorth>0</southNorth>
            <eastWest>0</eastWest>
            <aboveBelow>0</aboveBelow>
          </divisions>
          <horizontalObstacles>
            <gap>0</gap>
            <bridge>0</bridge>
            <door>0</door>
            <puzzle>0</puzzle>
            <jump>0</jump>
          </horizontalObstacles>
          <verticalObstacles>
            <stairs>0</stairs>
            <ladder>0</ladder>
            <jump>0</jump>
          </verticalObstacles>
          <hintLikelihood>1</hintLikelihood>
        </specification>
      </ClassroomDecorator> -->

      <DrawingDecorator>
        <!-- coordinates for cuboid are inclusive -->
        <!-- <DrawCuboid x1="-2" y1="46" z1="-2" x2="7" y2="50" z2="13" type="air" />            -->
        <!-- <DrawCuboid x1="-2" y1="45" z1="-2" x2="7" y2="45" z2="13" type="lava" />            -->
        <!-- <DrawCuboid x1="1"  y1="45" z1="1"  x2="3" y2="45" z2="17" type="sandstone" />       -->
        <!-- <DrawBlock x="4"  y="45" z="1" type="cobblestone" />     -->
        <!-- <DrawBlock x="4"  y="45" z="7" type="lapis_block" />      -->
        <!-- <DrawLine x1="5" y1="110" z1="0" x2="2" y2="230" z2="0" type="air"/> -->
        <!-- <DrawCuboid x1="-4" y1="106" z1="-4" x2="19" y2="106" z2="19" type="lava" /> -->
        <!-- <DrawCuboid x1="0" y1="107" z1="0" x2="15" y2="107" z2="15" type="stone" /> -->
        <DrawCuboid x1="-10" y1="108" z1="-10" x2="16" y2="108" z2="16" type="stone" />
        <!-- <DrawCuboid x1="-100" y1="108" z1="-100" x2="100" y2="110" z2="100" type="air" /> -->
        <DrawCuboid x1="0" y1="108" z1="0" x2="8" y2="110" z2="8" type="air" />
        <!-- <DrawBlock x="5" y="107" z="5" type="redstone_block"/> -->

        <DrawEntity x="'''+str(np.random.randint(1,8)+0.5)+'''" y="108.0" z="'''+str(np.random.randint(1,8)+0.5)+'''" type="Zombie" yaw="0" pitch="0" xVel="0" yVel="0" zVel="0"/>
        <!-- <DrawEntity x="13.5" y="108.0" z="8.5" type="Zombie" yaw="0" pitch="0" xVel="0" yVel="0" zVel="0"/> -->
      </DrawingDecorator>

      <!--
      <MazeDecorator>
        <SizeAndPosition length="20" width="20" xOrigin="0" yOrigin="215" zOrigin="410" height="180"/>
        <GapProbability variance="0.1">0.9</GapProbability>
        <Seed>random</Seed>
        <MaterialSeed>random</MaterialSeed>
        <AllowDiagonalMovement>false</AllowDiagonalMovement>
        <StartBlock fixedToEdge="true" type="emerald_block" height="1"/>
        <EndBlock fixedToEdge="true" type="redstone_block" height="12"/>
        <PathBlock type="glowstone" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="1"/>
        <FloorBlock type="air"/>
        <SubgoalBlock type="beacon"/>
        <GapBlock type="stained_hardened_clay" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="3"/>
      </MazeDecorator>
      -->

      <!--
      <ClassroomDecorator>
        <complexity>
          <building>0.5</building>
          <path>0.5</path>
          <division>1</division>
          <obstacle>1</obstacle>
          <hint>0</hint>
        </complexity>
      </ClassroomDecorator>
      -->

      <ServerQuitFromTimeUp timeLimitMs="40000" description="out_of_time"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

<!-- Can be Survival, Creative, Spectator -->
  <AgentSection mode="Survival"> 
    <Name>Steve</Name>
    <AgentStart>
      <!-- <Placement x="1.5" y="108.0" z="1.5" pitch="0" yaw="0"/> -->
      <Placement x="6.5" y="108.0" z="5.5" pitch="45" yaw="0"/>
      <Inventory>
        <!-- <InventoryItem slot="0" type="diamond_sword"/> -->
        <!-- <InventoryItem slot="0" type="stone" quantity="64"/> -->
        <InventoryItem slot="0" type="diamond_pickaxe"/>
        <InventoryItem slot="1" type="diamond_axe"/>
        <InventoryItem slot="7" type="stick" quantity="1"/>
        <InventoryItem slot="8" type="diamond" quantity="2"/>
      </Inventory>
    </AgentStart>
    
    <AgentHandlers>
      
      <DiscreteMovementCommands/>

      <SimpleCraftCommands/>

      <!-- <DiscreteMovementCommands> -->
        <!-- Can also have a deny-list -->
        <!-- <ModifierList type="allow-list">
          <command>move</command>
          <command>strafe</command>
          <command>turn</command>
          <command>look</command>
          <command>jumpmove</command>
          <command>attack</command>
        </ModifierList>
      </DiscreteMovementCommands> -->

      <!-- <ContinuousMovementCommands turnSpeedDegs="180"> -->
      <!-- <ContinuousMovementCommands> -->
        <!-- <ModifierList type="deny-list">
          <command>turn</command>
        </ModifierList> -->
        <!-- <ModifierList type="allow-list">
          <command>jump</command>
          <command>move</command>
          <command>strafe</command>
          <command>turn</command>
          <command>pitch</command>
        </ModifierList> -->
      <!-- </ContinuousMovementCommands> -->
      
      <!-- <ObservationFromFullStats/> -->

      <ObservationFromHotBar/>

      <!-- <ObservationFromRecentCommands/>  -->

      <!-- A little overkill -->
      <!-- <ObservationFromFullInventory/>  -->
      
      <ObservationFromGrid>
        <Grid name="floor9x9">
          <min x="-4" y="0" z="-4"/>
          <max x="4" y="0" z="4"/>
        </Grid>
      </ObservationFromGrid>

      <ObservationFromNearbyEntities>
        <Range name="nearby_entites" xrange="9" yrange="3" zrange="9"/>
      </ObservationFromNearbyEntities>

      <InventoryCommands/>

      <MissionQuitCommands/>

      <RewardForSendingCommand reward="-0.05" />
      
      <RewardForMissionEnd rewardForDeath="-1.0">
        <Reward description="found_goal" reward="1" />
        <Reward description="out_of_time" reward="-0.8" />
      </RewardForMissionEnd>
      
      <RewardForTouchingBlockType>
        <Block reward="-1.0" type="lava" behaviour="onceOnly"/>
        <Block reward="1.0" type="lapis_block" behaviour="onceOnly"/>
      </RewardForTouchingBlockType>

      <!-- 
      <VideoProducer want_depth="true">
        <Width>432</Width>
        <Height>240</Height>
      </VideoProducer> 
      -->

      <AgentQuitFromTouchingBlockType>
          <Block type="lava" description="death"/>
          <Block type="lapis_block" description="found_goal"/>
          <!-- <Block type="redstone_block" description="found_goal"/> -->
      </AgentQuitFromTouchingBlockType>

      <!-- 
      <AgentQuitFromReachingPosition>
        <Marker x="-26.5" y="40" z="0.5" tolerance="0.5" description="Goal_found"/>
      </AgentQuitFromReachingPosition> 
      -->

      <AgentQuitFromReachingCommandQuota description="command_quota_reached" total="10000"/>

      <AgentQuitFromCollectingItem>
        <Item type="diamond_sword" description="sword_aquired"/>
      </AgentQuitFromCollectingItem>

      <AgentQuitFromTimeUp timeLimitMs="30000" description="Steve survived!"/> 

    </AgentHandlers>
  </AgentSection>

</Mission>
'''