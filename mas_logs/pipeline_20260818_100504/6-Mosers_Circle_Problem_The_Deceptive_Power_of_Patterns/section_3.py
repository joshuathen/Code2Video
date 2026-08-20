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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Patterns look like doubling powers.", "Wait, check six points.", "The total is thirty-one."]
        self.setup_layout("The Reality Check: n=6", lecture_lines)
        
        # Use asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        self.place_at_grid(circle, "C3", scale_factor=0.8)
        self.add(circle)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        power_label = Text("2^(n-1)", color="#FF5733", font_size=24)
        self.place_at_grid(power_label, "A5", scale_factor=1.0)
        self.play(Write(power_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        n_label = Text("n = 6", color="#FF5733", font_size=24)
        # Apply fix from issue 27
        self.place_at_grid(n_label, "D5", scale_factor=0.7)
        self.play(Write(n_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        # Apply fix from issue 26
        actual_val = Text("Regions = 31", color="#33FF57", font_size=24)
        self.place_at_grid(actual_val, "C5", scale_factor=0.6)
        
        divergence_indicator = Circle(radius=0.2, color="#FF33FF", fill_opacity=0.5)
        self.place_at_grid(divergence_indicator, "C6", scale_factor=0.5)
        
        # Apply fix from issue 28
        pattern_broken = Text("Pattern Broken", color="#FF33FF", font_size=30)
        self.place_in_area(pattern_broken, "E4", "F6", scale_factor=0.8)
        
        self.play(
            Write(actual_val),
            FadeIn(divergence_indicator),
            Indicate(actual_val),
            Write(pattern_broken)
        )
        self.play(Flash(divergence_indicator.get_center()))
        self.wait(2)
