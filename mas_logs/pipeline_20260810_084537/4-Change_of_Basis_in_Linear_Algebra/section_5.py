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
        lecture_lines = [
            "Basis is just a coordinate system choice.",
            "Transition matrices act as system translators.",
            "Perspective changes, but the space remains invariant."
        ]
        self.setup_layout("Summary & Reflection", lecture_lines)
        
        # Assets
        perspective_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/perspective.svg")
        coord_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coordinate.svg")
        
        recap_label = VGroup(Text("Changing Perspectives", color=YELLOW, font_size=24), perspective_icon).arrange(DOWN)
        transition_matrix = MathTex("P", color=RED).scale(2)
        coord_map = VGroup(coord_icon).set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(recap_label, 'B4', scale_factor=0.9)
        self.play(FadeIn(recap_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        self.place_at_grid(transition_matrix, 'D4', scale_factor=0.85)
        self.play(FadeIn(transition_matrix))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.place_in_area(coord_map, 'E2', 'F4', scale_factor=0.75)
        self.play(FadeIn(coord_map))
        self.wait(2)
