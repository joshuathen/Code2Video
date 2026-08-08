from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetch data from storyboard
        title_text = "The Impossible Coincidence"
        lecture_lines = [
            "Two blocks and a wall, on a frictionless floor.",
            "A massive block slides toward a smaller one.",
            "We count every collision: block-to-block and block-to-wall.",
            "With mass ratio 100, we count exactly 31 collisions.",
            "Amazingly, increasing mass reveals the digits of Pi!"
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors from animation plan
        COLOR_M = "#FF4500" # Block M
        COLOR_m = "#ADD8E6" # Block m
        COLOR_FLASH = "#FFFF00"
        COLOR_PI = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # "Two blocks and a wall, on a frictionless floor."
        self.lecture[0].set_color(WHITE)
        
        # Grid layout
        # Floor: Line along Row E, Cols 1 to 6
        floor = Line(self.grid["E1"] + LEFT*0.5, self.grid["E6"] + RIGHT*0.5, color=WHITE)
        # Wall: Vertical line at Col 2
        wall_x = self.grid["E2"][0] - 0.5
        wall = Line(
            [wall_x, self.grid["B2"][1], 0],
            [wall_x, self.grid["E2"][1], 0],
            stroke_width=8,
            color=GRAY
        )
        
        # Blocks
        m_block = Square(side_length=0.4, fill_opacity=1, color=COLOR_m)
        self.place_at_grid(m_block, "E3")
        m_block.shift(UP * 0.2) # Sit on floor (Row E y = -1.8)
        
        M_block = Square(side_length=0.8, fill_opacity=1, color=COLOR_M)
        self.place_at_grid(M_block, "E6") # Updated to E6 per Issue 39
        M_block.shift(UP * 0.4) # Sit on floor
        
        self.play(Create(floor), Create(wall), FadeIn(m_block), FadeIn(M_block))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A massive block slides toward a smaller one."
        self.lecture[1].set_color(COLOR_M)
        
        # Collision Counter setup
        collision_count = 0
        counter_label = Text("Collisions:", font_size=24, color=WHITE)
        counter_num = Integer(collision_count, font_size=24, color=WHITE)
        counter_group = VGroup(counter_label, counter_num).arrange(RIGHT, buff=0.2)
        self.place_at_grid(counter_group, "B5") # Updated to B5 per Issue 38
        
        self.play(Write(counter_group))
        
        # Physics Parameters
        m = 1
        M = 100
        v_m = 0
        v_M = -3.0 # Initial velocity of large block
        
        # Positions relative to the coordinate system
        x_m = m_block.get_center()[0]
        x_M = M_block.get_center()[0]
        w_m = 0.4
        w_M = 0.8
        
        # === Animation for Lecture Line 3 ===
        # "We count every collision: block-to-block and block-to-wall."
        self.lecture[2].set_color(COLOR_FLASH)
        
        total_collisions = 31
        
        while collision_count < total_collisions:
            # Time to next event
            # 1. m-M collision (if v_M < v_m)
            if v_M < v_m:
                t_collision = (x_M - x_m - (w_m/2 + w_M/2)) / (v_m - v_M)
            else:
                t_collision = float('inf')
                
            # 2. m-wall collision (if v_m < 0)
            if v_m < 0:
                t_wall = (x_m - wall_x - w_m/2) / (-v_m)
            else:
                t_wall = float('inf')
            
            # Select next event
            if t_collision < t_wall:
                dt = t_collision
                collision_type = "block"
                new_v_m = ((m - M) / (m + M)) * v_m + (2 * M / (m + M)) * v_M
                new_v_M = (2 * m / (m + M)) * v_m + ((M - m) / (m + M)) * v_M
            else:
                dt = t_wall
                collision_type = "wall"
                new_v_m = -v_m
                new_v_M = v_M
            
            # Adjust animation speed - faster for earlier, slower/smaller steps later if needed
            # For 31 collisions, we can keep it relatively snappy
            rt = max(0.03, 0.15 * (0.94 ** collision_count))
            
            # Animate movement
            self.play(
                m_block.animate.shift(RIGHT * v_m * dt),
                M_block.animate.shift(RIGHT * v_M * dt),
                run_time=rt,
                rate_func=linear
            )
            
            # Update state
            x_m += v_m * dt
            x_M += v_M * dt
            v_m, v_M = new_v_m, new_v_M
            collision_count += 1
            counter_num.set_value(collision_count)
            
            # Flash at collision point
            if collision_type == "block":
                flash_pos = m_block.get_right()
            else:
                flash_pos = m_block.get_left()
            
            # Play flash
            self.play(
                Flash(flash_pos, color=COLOR_FLASH, flash_radius=0.15, line_length=0.1),
                run_time=min(0.05, rt)
            )

        # === Animation for Lecture Line 4 ===
        # "With mass ratio 100, we count exactly 31 collisions."
        self.lecture[3].set_color(COLOR_m)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Amazingly, increasing mass reveals the digits of Pi!"
        self.lecture[4].set_color(COLOR_PI)
        
        pi_val = MathTex(r"\pi \approx 3.141...", color=COLOR_PI)
        self.place_at_grid(pi_val, "C5", scale_factor=1.0) # Updated to C5 and scale 1.0 per Issue 37
        
        final_result = Text(f"Total Collisions: {collision_count}", font_size=26, color=COLOR_PI)
        self.place_at_grid(final_result, "D5") # Updated to D5 per Issue 38
        
        self.play(FadeIn(pi_val), Write(final_result))
        self.wait(3)
