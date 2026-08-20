from manim import *
import numpy as np

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
        lecture_lines = [
            "Energy spectrum E(k) distributes energy across scales.",
            "Large eddies act as low-frequency bass.",
            "Small eddies behave as high-frequency treble.",
            "The power-law follows a negative five-thirds slope.",
            "This slope confirms energy conservation in transfer."
        ]
        self.setup_layout("The -5/3 Law: Spectral Analysis", lecture_lines)
        
        # --- Create Axes ---
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True}
        )
        axes_labels = axes.get_axis_labels(x_label="k", y_label="E(k)")
        spectrum_group = VGroup(axes, axes_labels)
        self.place_in_area(spectrum_group, 'C3', 'F5', scale_factor=0.6)
        
        # --- Create Power Law curve ---
        graph = axes.plot(lambda x: 10 * (x + 0.1)**(-5/3), x_range=[0.1, 9], color=BLUE)
        
        # --- Assets ---
        bass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bass.svg")
        treble_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/treble.svg")
        
        # --- Animation for Lecture Line 1 ---
        self.lecture_lines_objects[0].set_color(YELLOW)
        self.play(Create(spectrum_group), Create(graph))
        self.wait(1)

        # --- Animation for Lecture Line 2 ---
        self.lecture_lines_objects[0].set_color(WHITE)
        self.lecture_lines_objects[1].set_color(YELLOW)
        bass_label = Text("Bass", font_size=20)
        self.place_at_grid(bass_label, 'C2', scale_factor=0.7)
        self.place_at_grid(bass_icon, 'B2', scale_factor=0.5)
        self.play(FadeIn(bass_icon), Write(bass_label))
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        self.lecture_lines_objects[1].set_color(WHITE)
        self.lecture_lines_objects[2].set_color(YELLOW)
        treble_label = Text("Treble", font_size=20)
        self.place_at_grid(treble_label, 'F5', scale_factor=0.7)
        self.place_at_grid(treble_icon, 'F6', scale_factor=0.5)
        self.play(FadeIn(treble_icon), Write(treble_label))
        self.wait(1)

        # --- Animation for Lecture Line 4 ---
        self.lecture_lines_objects[2].set_color(WHITE)
        self.lecture_lines_objects[3].set_color(YELLOW)
        slope_line = Line(axes.c2p(1, 4), axes.c2p(5, 1), color="#FF4500", stroke_width=4)
        power_law_annotation = MathTex("-5/3", color="#FF4500", font_size=24)
        self.place_at_grid(power_law_annotation, 'F4', scale_factor=0.65)
        self.play(Create(slope_line), Write(power_law_annotation))
        self.wait(1)

        # --- Animation for Lecture Line 5 ---
        self.lecture_lines_objects[3].set_color(WHITE)
        self.lecture_lines_objects[4].set_color(YELLOW)
        self.play(Indicate(slope_line))
        self.wait(1)
