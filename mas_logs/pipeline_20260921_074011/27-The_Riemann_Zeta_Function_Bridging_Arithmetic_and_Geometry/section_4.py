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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Riemann Hypothesis: The Critical Strip", [
            "The critical strip lies between 0 and 1.",
            "The hypothesis concerns where the function equals zero.",
            "All non-trivial zeros lie on line 1/2."
        ])
        
        # Plotting container
        axes = Axes(x_range=[-1, 3, 1], y_range=[-2, 2, 1], axis_config={"include_numbers": False}).scale(0.5)
        critical_strip = Rectangle(
            width=axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0],
            height=axes.c2p(0, 2)[1] - axes.c2p(0, -2)[1],
            fill_opacity=0.2, fill_color=BLUE, stroke_width=0
        ).move_to(axes.c2p(0.5, 0))
        
        critical_line = Line(axes.c2p(0.5, -2), axes.c2p(0.5, 2), color="#00FFFF", stroke_width=4)
        zeros = VGroup(*[Dot(axes.c2p(0.5, i * 0.5), color=YELLOW) for i in range(-3, 4) if i != 0])
        
        riemann_plot = VGroup(axes, critical_strip, critical_line, zeros)
        self.place_in_area(riemann_plot, 'D2', 'F6', scale_factor=0.9)
        
        # Remove individual elements from scene to add them sequentially
        self.remove(axes, critical_strip, critical_line, zeros)

        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), FadeIn(critical_strip))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(zeros, 'C4', scale_factor=0.6)
        self.play(FadeIn(zeros))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(Create(critical_line))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
