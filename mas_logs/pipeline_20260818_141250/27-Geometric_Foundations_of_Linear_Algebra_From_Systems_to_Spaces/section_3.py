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
        lecture_lines = [
            "Column space is the span of column vectors.",
            "It defines all reachable destinations in space.",
            "Think of a 3D printer's reachable range.",
            "Vectors can generate a line or plane.",
            "Every reachable point lies within this span."
        ]
        self.setup_layout("Column Space: The Reachable World", lecture_lines)

        # Assets
        printer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/printer.svg")
        
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3])
        vec1 = Arrow(ORIGIN, [1, 2, 1], color=WHITE, buff=0)
        vec2 = Arrow(ORIGIN, [2, -1, 0], color=WHITE, buff=0)
        vector_labels = VGroup(MathTex("v_1"), MathTex("v_2")).arrange(RIGHT)
        
        # Grid area for 3D axes
        self.place_at_grid(axes, 'D3', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(axes), Create(vec1), Create(vec2))
        self.place_at_grid(vector_labels, 'D4', scale_factor=0.4)
        self.play(FadeIn(vector_labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Show span (Plane)
        spanned_plane = Polygon(ORIGIN, [1, 2, 1], [3, 1, 1], [2, -1, 0], color="#00FF00", fill_opacity=0.3)
        self.place_in_area(spanned_plane, 'D2', 'F5', scale_factor=0.5)
        self.play(FadeIn(spanned_plane))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_at_grid(printer_icon, 'B3', scale_factor=0.5)
        self.play(FadeIn(printer_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        target_point = Dot(color=YELLOW).move_to(axes.c2p(1, 1, 0.5))
        self.play(FadeIn(target_point))
        self.wait(2)
