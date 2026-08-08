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
        # Setup layout
        self.setup_layout("The Setup: A Strange Observation", [
            "Imagine two blocks on a frictionless surface.",
            "A small block sits near a vertical wall.",
            "A large block slides in from the right."
        ])
        
        # Define colors for lecture lines
        line_colors = [WHITE, GREEN, BLUE]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(line_colors[0]))
        
        # Floor and wall positioning based on grid
        # wall_x: x-coordinate of the vertical wall (left side of Col 1 area)
        # floor_y: y-coordinate of the floor (bottom side of Row D area)
        wall_x = 0.1 
        floor_y = self.grid["D1"][1] - 0.4
        
        wall = Line(
            np.array([wall_x, self.grid["A1"][1], 0]),
            np.array([wall_x, floor_y, 0]),
            color=WHITE, stroke_width=4
        )
        floor = Line(
            np.array([wall_x, floor_y, 0]),
            np.array([6.5, floor_y, 0]),
            color=WHITE, stroke_width=4
        )
        
        self.play(Create(wall), Create(floor))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(line_colors[1]))
        
        # Create green block group (m)
        # [Fix for Issue 22]: Anchor m_block at D2 with scale_factor=1.0
        m_block = Square(side_length=0.8, color=GREEN, fill_opacity=0.8)
        self.place_at_grid(m_block, "D2", scale_factor=1.0)
        m_label = MathTex("m", color=GREEN).scale(0.8)
        m_label.next_to(m_block, UP, buff=0.1)
        m_group = VGroup(m_block, m_label)
        
        self.play(FadeIn(m_group))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(line_colors[2]))
        
        # Create blue block group (M)
        # [Fix for Issue 21]: Anchor M_block at D6 with scale_factor=1.5
        M_block = Square(side_length=0.8, color=BLUE, fill_opacity=0.8)
        self.place_at_grid(M_block, "D6", scale_factor=1.5)
        M_label = MathTex("M", color=BLUE).scale(1.0) # Scaled up slightly to match block
        M_label.next_to(M_block, UP, buff=0.1)
        M_group = VGroup(M_block, M_label)
        
        # Collision Counter
        # [Fix for Issue 20]: Move counter_group to A6 with scale_factor=0.9
        collision_count = 0
        counter_label = Text("Collisions: ", font_size=20, color=WHITE)
        counter_val = Integer(collision_count, font_size=20, color=WHITE)
        counter_group = VGroup(counter_label, counter_val).arrange(RIGHT, buff=0.1)
        self.place_at_grid(counter_group, "A6", scale_factor=0.9)
        
        self.play(FadeIn(M_group), FadeIn(counter_group))
        self.wait(0.5)
        
        # Movement and collision logic (Case: M = m)
        # Note: M_block scale=1.5 means width is 0.8 * 1.5 = 1.2.
        # m_block scale=1.0 means width is 0.8 * 1.0 = 0.8.
        # Center of D6 is at x=5.5. Center of D2 is at x=1.5.
        
        # 1. M moves to hit m
        # M_group center starts at 5.5. m_group center is at 1.5.
        # Touch point center_x_M = center_x_m + width_m/2 + width_M/2 = 1.5 + 0.4 + 0.6 = 2.5
        # Distance to shift M = 5.5 - 2.5 = 3.0
        self.play(
            M_group.animate.shift(LEFT * 3.0),
            run_time=1.5, rate_func=linear
        )
        
        # Collision 1 (M hits m)
        collision_count += 1
        counter_val.set_value(collision_count)
        self.play(counter_val.animate.scale(1.3).set_color(YELLOW), run_time=0.1)
        self.play(counter_val.animate.scale(1/1.3).set_color(WHITE), run_time=0.1)
        
        # 2. m moves to wall
        # m_group center starts at 1.5. Wall is at x=0.1.
        # Touch point center_x_m = wall_x + width_m/2 = 0.1 + 0.4 = 0.5
        # Distance to shift m = 1.5 - 0.5 = 1.0
        self.play(
            m_group.animate.shift(LEFT * 1.0),
            run_time=0.6, rate_func=linear
        )
        
        # Collision 2 (m hits wall)
        collision_count += 1
        counter_val.set_value(collision_count)
        self.play(counter_val.animate.scale(1.3).set_color(YELLOW), run_time=0.1)
        self.play(counter_val.animate.scale(1/1.3).set_color(WHITE), run_time=0.1)
        
        # 3. m moves back to M
        # m_group center starts at 0.5. M_group center is at 2.5.
        # Touch point center_x_m = center_x_M - width_M/2 - width_m/2 = 2.5 - 0.6 - 0.4 = 1.5
        # Distance to shift m = 1.5 - 0.5 = 1.0
        self.play(
            m_group.animate.shift(RIGHT * 1.0),
            run_time=0.6, rate_func=linear
        )
        
        # Collision 3 (m hits M)
        collision_count += 1
        counter_val.set_value(collision_count)
        self.play(counter_val.animate.scale(1.3).set_color(YELLOW), run_time=0.1)
        self.play(counter_val.animate.scale(1/1.3).set_color(WHITE), run_time=0.1)
        
        # 4. M moves away
        # M_group center starts at 2.5. Off-screen is around 6.5.
        # Distance to shift M = 6.5 - 2.5 = 4.0
        self.play(
            M_group.animate.shift(RIGHT * 4.0),
            run_time=1.5, rate_func=linear
        )
        
        self.wait(2)
