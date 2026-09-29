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
        self.setup_layout("Summary and Open Frontiers", [
            "The square problem remains open.", 
            "Fractals challenge current proofs.", 
            "Can we find a square there?"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Revisit the initial curve and fitted square using asset
        curve = ParametricFunction(
            lambda t: np.array([1.5 * np.cos(t) + 0.5 * np.cos(3 * t), 1.5 * np.sin(t) - 0.5 * np.sin(3 * t), 0]),
            t_range=[0, 2 * PI]
        ).set_color("#00FFFF")
        
        # Asset usage
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg").set_color("#00FFFF")
        
        content1 = VGroup(curve, square)
        self.place_in_area(content1, "A2", "C4", scale_factor=0.5)
        self.play(Create(curve), FadeIn(square))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Recap the key steps: topology, mapping, and existence
        self.lecture[1].set_color("#FF00FF")
        steps = VGroup(
            Text("1. Topology", font_size=20),
            Text("2. Mapping", font_size=20),
            Text("3. Existence", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(steps, "D2", scale_factor=0.7)
        self.play(Write(steps))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Mention other related problems in geometry and topology
        self.lecture[2].set_color("#FFFF00")
        question = Text("Can we find a square there?", font_size=24, color="#FFFF00")
        self.place_at_grid(question, "D5", scale_factor=0.7)
        self.play(FadeIn(question))
        self.wait(2)
