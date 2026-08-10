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
        self.setup_layout("Introduction: What is a Vector?", [
            "Vectors are arrows, not just lists of numbers.",
            "Think of an arrow from origin to (3, 2).",
            "The starting point is always the origin.",
            "The tip shows the destination point clearly.",
            "This arrow defines a specific movement in space."
        ])

        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        # Grid setup (Fixed as per Critic)
        axes = Axes(x_range=[-1, 5], y_range=[-1, 4], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(compass, 'A4', scale_factor=0.3)
        point = Dot(axes.c2p(0, 0), color=WHITE)
        self.play(FadeIn(compass), FadeIn(point))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        vector = Arrow(axes.c2p(0, 0), axes.c2p(3, 2), buff=0, color=BLUE)
        self.play(Create(vector))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        v_label = MathTex(r"\\vec{v}", color="#FF00FF")
        self.place_at_grid(v_label, 'C4', scale_factor=0.7)
        self.play(Write(v_label))
        self.lecture[2].set_color("#FF00FF")

        # === Animation for Lecture Line 4 ===
        x_line = DashedLine(axes.c2p(0, 0), axes.c2p(3, 0), color=RED)
        y_line = DashedLine(axes.c2p(3, 0), axes.c2p(3, 2), color=GREEN)
        guide_lines = VGroup(x_line, y_line)
        self.place_in_area(guide_lines, 'B3', 'D5', scale_factor=0.7)
        self.place_at_grid(ruler, 'E5', scale_factor=0.3)
        self.play(Create(guide_lines), FadeIn(ruler))
        self.lecture[3].set_color(ORANGE)

        # === Animation for Lecture Line 5 ===
        self.play(vector.animate.set_color(YELLOW))
        self.lecture[4].set_color(YELLOW)
        self.wait(1)
