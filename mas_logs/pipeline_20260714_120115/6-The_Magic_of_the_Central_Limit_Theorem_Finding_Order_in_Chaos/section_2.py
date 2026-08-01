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
        # Lecture lines synchronized with script updates
        lecture_lines = [
            "Stat-Bot selects five random members from our population.",
            "We calculate the average value for this small sample.",
            "This average becomes a single point in our bin.",
            "We repeat this sampling process hundreds of times.",
            "The resulting collection is the Sampling Distribution of the Mean."
        ]
        self.setup_layout("The Sampling Experiment: Taking Averages", lecture_lines)

        # Colors
        STAT_BOT_COLOR = "#58C4DD"
        HIGHLIGHT_COLOR = "#FF8080"
        POP_COLOR = "#FFFFFF"
        BIN_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Visual: Stat-Bot selects five random members from our population.
        self.lecture[0].set_color(STAT_BOT_COLOR)
        
        # Population dots (Visual Anchor for 'Population')
        # Fix from Issue 48: place_in_area(population, 'A3', 'B5', scale_factor=1.0)
        population = VGroup(*[Dot(radius=0.06, color=POP_COLOR) for _ in range(25)])
        self.place_in_area(population, "A3", "B5", scale_factor=1.0)
        # Randomize positions within the area
        for dot in population:
            dot.shift(np.random.uniform(-0.8, 0.8) * RIGHT + np.random.uniform(-0.4, 0.4) * UP)
        self.add(population)
        
        # Stat-Bot (Visual Anchor for 'Robot Collector')
        # Issue 34: Use SVG asset
        # Fix from Issue 48: place_at_grid(stat_bot, 'A2', scale_factor=1.2)
        stat_bot = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg")
        stat_bot.set_color(STAT_BOT_COLOR)
        self.place_at_grid(stat_bot, "A2", scale_factor=1.2)
        self.play(FadeIn(stat_bot))
        
        # Picking 5 dots - Highlighting the sample
        selected_indices = [2, 7, 12, 17, 22]
        selected_dots = VGroup(*[population[i] for i in selected_indices])
        
        # Move Stat-Bot to population and highlight the sample
        self.play(
            stat_bot.animate.move_to(self.grid["A4"]),
            selected_dots.animate.set_color(HIGHLIGHT_COLOR).scale(1.5),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: We calculate the average value for this small sample.
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Fix from Issue 47: place_in_area(avg_math, 'C2', 'C5', scale_factor=0.8)
        avg_math = Text("Sample Average (x̄) = 168", color=HIGHLIGHT_COLOR)
        self.place_in_area(avg_math, "C2", "C5", scale_factor=0.8)
        
        self.play(Write(avg_math))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: This average becomes a single point in our bin.
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Create a visual container (bin) for the sampling distribution
        # Located in rows E-F (columns 2-5)
        bin_lines = VGroup(
            Line(self.grid["F2"], self.grid["F5"], color=BIN_COLOR, stroke_width=3), # Bottom
            Line(self.grid["E2"], self.grid["F2"], color=BIN_COLOR, stroke_width=3), # Left side
            Line(self.grid["E5"], self.grid["F5"], color=BIN_COLOR, stroke_width=3)  # Right side
        )
        self.play(Create(bin_lines))
        
        # Transform the selection/average into a single data point (dot)
        dot_from_avg = Dot(color=HIGHLIGHT_COLOR, radius=0.12)
        dot_from_avg.move_to(avg_math.get_center())
        
        self.play(
            FadeOut(avg_math),
            ReplacementTransform(selected_dots.copy(), dot_from_avg),
            run_time=1.5
        )
        
        # Drop the dot into the bin
        bin_center_bottom = self.grid["F3"] + RIGHT * 0.5 
        self.play(
            dot_from_avg.animate.move_to(bin_center_bottom + UP*0.15),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Visual: We repeat this sampling process hundreds of times.
        self.lecture[3].set_color(POP_COLOR)
        
        # Fast sampling simulation to show volume of data accumulating in the bin
        sim_dots = VGroup()
        for _ in range(100):
            d = Dot(radius=0.07, color=HIGHLIGHT_COLOR)
            # Randomized spread within the bin boundaries (columns 2 to 5)
            # Center of bin is around self.grid["F3"] + 0.5*RIGHT
            offset_x = np.random.uniform(-1.4, 1.4)
            offset_y = np.random.uniform(0.1, 0.8)
            d.move_to(bin_center_bottom + RIGHT*offset_x + UP*offset_y)
            sim_dots.add(d)
            
        self.play(
            LaggedStart(*[FadeIn(d, run_time=0.05) for d in sim_dots], lag_ratio=0.03),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Visual: The resulting collection is the Sampling Distribution of the Mean.
        self.lecture[4].set_color(WHITE)
        
        # Final label for the new distribution
        # Fix from Issue 47: place_in_area(final_label, 'C2', 'C5', scale_factor=0.8)
        final_label = Text("Sampling Distribution\nof the Mean", font_size=24, color=WHITE)
        self.place_in_area(final_label, "C2", "C5", scale_factor=0.8)
        
        self.play(Write(final_label))
        self.wait(2)
