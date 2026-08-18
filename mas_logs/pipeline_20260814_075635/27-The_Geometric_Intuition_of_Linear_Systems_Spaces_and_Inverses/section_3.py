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
        self.setup_layout("Null Space: The 'Ghost' Inputs", [
            "Null space contains vectors reaching zero.",
            "These inputs effectively vanish.",
            "Non-zero null space implies non-uniqueness."
        ])
        
        # Assets
        ghost = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ghost.svg")
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], axis_config={"include_tip": True}).scale(0.6)
        self.place_at_grid(axes, "D4")
        
        input_vectors = VGroup(
            Arrow(start=axes.c2p(0, 0), end=axes.c2p(1, 1), color=BLUE),
            Arrow(start=axes.c2p(0, 0), end=axes.c2p(-1, 0.5), color=BLUE),
            Arrow(start=axes.c2p(0, 0), end=axes.c2p(0.5, -1), color=BLUE)
        )
        self.add(input_vectors)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        target_vecs = VGroup(
            Arrow(start=axes.c2p(0, 0), end=axes.c2p(0, 0), color=RED),
            Arrow(start=axes.c2p(0, 0), end=axes.c2p(0, 0), color=RED),
            Arrow(start=axes.c2p(0, 0), end=axes.c2p(0, 0), color=RED)
        )
        # Position ghost at the center (origin of the transformation)
        self.place_at_grid(ghost, "D4", scale_factor=0.3)
        self.play(ReplacementTransform(input_vectors, target_vecs), FadeIn(ghost))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeOut(target_vecs), ghost.animate.set_opacity(0.2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        null_space_line = Line(axes.c2p(-2, 1), axes.c2p(2, -1), color="#FF33A8", stroke_width=4)
        self.place_in_area(null_space_line, "D4", "F6", scale_factor=0.7)
        self.play(Create(null_space_line))
        self.wait(1)
