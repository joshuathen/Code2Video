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
        lecture_lines = [
            "Increase mass ratio drastically.",
            "Collisions approach Pi power.",
            "The wedge angle shrinks.",
            "Trajectory fits the wedge.",
            "This reflects Pi digits."
        ]
        self.setup_layout("Unveiling Pi: The Arc and the Ratio", lecture_lines)
        
        # Load Assets
        wedge_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wedge.svg"
        wedge_icon = SVGMobject(wedge_asset)
        
        # Elements
        arc = Arc(radius=1.5, start_angle=0, angle=PI/4, color="#FFCC00")
        radius_line = Line(ORIGIN, 1.5 * RIGHT, color="#FFFFFF")
        ratio_label = Text("Mass Ratio: 10^n", font_size=24, color="#AAAAAA")
        highlight_segment = Arc(radius=1.5, start_angle=0, angle=PI/8, color="#FF0000", stroke_width=8)
        pi_text = Text("π ≈ 3.14159", font_size=36, color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCC00"))
        # Using SVGMobject wedge as reference for the arc
        self.place_in_area(arc, 'B2', 'D4', scale_factor=1.0)
        wedge_icon.replace(arc)
        self.play(Create(arc), FadeIn(wedge_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(radius_line, 'C3')
        self.play(Create(radius_line))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#AAAAAA"))
        self.place_at_grid(ratio_label, 'A5', scale_factor=0.9)
        self.play(Write(ratio_label))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.place_in_area(highlight_segment, 'B2', 'D4', scale_factor=1.0)
        self.play(Create(highlight_segment))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.place_at_grid(pi_text, 'D3', scale_factor=1.0)
        self.play(FadeIn(pi_text))
        
        self.wait(2)
