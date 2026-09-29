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
        self.setup_layout("Decoding the Winding Number", [
            "Winding numbers count wraps around the origin.",
            "They act as topological root fingerprints.",
            "This connects geometry to the Argument Principle."
        ])
        
        # Elements
        root = Dot(color=RED)
        self.place_in_area(root, 'B3', 'C4', scale_factor=0.6)
        
        # SVG asset
        fingerprint = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fingerprint.svg")
        fingerprint.set_color(WHITE)
        
        curve = ParametricFunction(
            lambda t: np.array([
                2 * np.cos(t) + 0.5 * np.cos(3*t),
                2 * np.sin(t) + 0.5 * np.sin(3*t),
                0
            ]), t_range=[0, 2*PI], color=WHITE
        )
        
        # Wrap the whole graphic group
        graphic_group = VGroup(root, curve, fingerprint)
        self.place_in_area(graphic_group, 'B4', 'E6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(root), Create(curve), FadeIn(fingerprint))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        mapping = curve.copy().set_color("#FFFF00")
        self.play(Transform(curve.copy(), mapping))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        counter = Text("W = 1", font_size=36, color="#00FF00")
        self.place_at_grid(counter, 'D4', scale_factor=1.2)
        self.play(Write(counter))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
