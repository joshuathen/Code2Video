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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Strange Phenomenon", [
            "For equal masses, we count exactly three collisions.",
            "If M is 100 times m, we get thirty-one.",
            "These counts remarkably match the digits of Pi."
        ])

        # Setup persistent environment
        floor = Line(self.grid["F1"], self.grid["F6"], color=WHITE)
        wall = Line(self.grid["A1"], self.grid["F1"], color=WHITE)
        self.add(floor, wall)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Mass label and Collision counter
        mass_label = MathTex("M = m", color="#00BFFF")
        self.place_at_grid(mass_label, "A5", scale_factor=0.8)
        
        counter_val = DecimalNumber(0, color=WHITE, num_decimal_places=0)
        counter_label = Text("Collisions:", font_size=24, color=WHITE)
        counter_group = VGroup(counter_label, counter_val).arrange(RIGHT, buff=0.2)
        self.place_at_grid(counter_group, "A2", scale_factor=0.8)
        
        # Blocks
        block_m = Square(side_length=0.6, fill_opacity=1, color=BLUE_E)
        block_M = Square(side_length=0.6, fill_opacity=1, color=BLUE_B)
        self.place_at_grid(block_m, "E2")
        self.place_at_grid(block_M, "E5")
        
        label_m = MathTex("m", font_size=24).add_updater(lambda m: m.move_to(block_m.get_center()))
        label_M = MathTex("M", font_size=24).add_updater(lambda m: m.move_to(block_M.get_center()))
        
        self.play(
            FadeIn(counter_group), 
            FadeIn(mass_label), 
            FadeIn(block_m), 
            FadeIn(block_M), 
            Write(label_m), 
            Write(label_M)
        )
        
        # 1. M hits m
        self.play(block_M.animate.move_to(self.grid["E2"] + RIGHT*0.6), run_time=1, rate_func=linear)
        counter_val.set_value(1)
        self.play(Flash(self.grid["E2"] + RIGHT*0.3, color=WHITE, flash_radius=0.3), run_time=0.2)
        
        # 2. m hits wall
        self.play(block_m.animate.move_to(self.grid["E1"] + RIGHT*0.3), run_time=0.8, rate_func=linear)
        counter_val.set_value(2)
        self.play(Flash(self.grid["E1"], color=WHITE, flash_radius=0.3), run_time=0.2)
        
        # 3. m hits M
        self.play(block_m.animate.move_to(self.grid["E2"] + RIGHT*0.6), run_time=0.8, rate_func=linear)
        counter_val.set_value(3)
        self.play(Flash(self.grid["E2"] + RIGHT*0.3, color=WHITE, flash_radius=0.3), run_time=0.2)
        
        # Final separation
        self.play(
            block_M.animate.shift(RIGHT * 2),
            block_m.animate.shift(RIGHT * 0.2),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        new_mass_label = MathTex("M = 100m", color="#00BFFF")
        self.place_at_grid(new_mass_label, "A5", scale_factor=0.8)
        
        # Reset and scale up
        self.play(
            Transform(mass_label, new_mass_label),
            counter_val.animate.set_value(0),
            block_m.animate.move_to(self.grid["E2"]),
            block_M.animate.move_to(self.grid["E5"]),
            run_time=1
        )
        self.play(block_M.animate.scale(1.4), run_time=0.5)
        
        # Quickly show count reaching 31
        self.play(ChangeDecimalToValue(counter_val, 31), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        final_mass_label = MathTex("M = 10,000m", color="#00BFFF")
        self.place_at_grid(final_mass_label, "A5", scale_factor=0.8)
        
        self.play(
            Transform(mass_label, final_mass_label),
            counter_val.animate.set_value(0),
            run_time=1
        )
        
        # Scale M even more
        self.play(block_M.animate.scale(1.3), run_time=0.5)
        
        # Count to 314
        self.play(ChangeDecimalToValue(counter_val, 314), run_time=2.5)
        
        # Pi highlighting
        pi_approx = MathTex(r"\pi \approx 3.14", color="#FFD700")
        self.place_at_grid(pi_approx, "B2", scale_factor=0.8)
        
        rect = SurroundingRectangle(counter_val, color=YELLOW, buff=0.1)
        self.play(Create(rect), Write(pi_approx))
        self.wait(2)
