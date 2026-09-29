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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Reflection", [
            "Uncertainty is inherent to wave motion.",
            "It is not a sensor limitation.",
            "Observation through waves has a tax."
        ])
        
        # Load asset
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        
        # === Animation for Lecture Line 1 ===
        # Summarize the trade-off with a simple 2x2 grid.
        rect1 = Rectangle(width=2, height=2, color=BLUE).set_fill(BLUE, opacity=0.3)
        icon1 = sensor_icon.copy()
        label1 = Text("Time Domain", font_size=18).next_to(rect1, UP)
        group1 = VGroup(rect1, icon1, label1)
        self.place_in_area(group1, 'B2', 'C3', scale_factor=1.2)
        self.play(FadeIn(group1))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Recap the key concept: 'Time-Frequency Duality'.
        rect2 = Rectangle(width=2, height=2, color=YELLOW).set_fill(YELLOW, opacity=0.3)
        icon2 = sensor_icon.copy()
        label2 = Text("Frequency Domain", font_size=18).next_to(rect2, UP)
        group2 = VGroup(rect2, icon2, label2)
        self.place_in_area(group2, 'B4', 'C5', scale_factor=1.2)
        self.play(FadeIn(group2))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Display final text: 'Precision depends on your goal.'
        final_text = Text("Precision depends on your goal.", font_size=24, color=GREEN)
        icon3 = sensor_icon.copy().scale(0.5).next_to(final_text, LEFT)
        final_group = VGroup(final_text, icon3)
        self.place_at_grid(final_group, 'E3', scale_factor=1.0)
        self.play(Write(final_text), FadeIn(icon3))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
