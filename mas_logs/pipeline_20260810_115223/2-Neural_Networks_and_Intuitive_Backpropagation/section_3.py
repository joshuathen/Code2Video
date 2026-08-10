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
            "Backpropagation is simply the Chain Rule.",
            "Distribute the Error backward across weights.",
            "Calculate each weight's contribution to error.",
            "Identify which connection requires adjustment.",
            "[Asset: network_propagation] shows error flowing backwards."
        ]
        self.setup_layout("Intuitive Backpropagation (The Credit Assignment)", lecture_lines)
        
        # Load the asset
        network_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg"
        network = SVGMobject(network_asset).scale(0.5)

        # Mobjects for animations
        derivs = MathTex(r"\frac{\partial E}{\partial w}").set_color("#00FFFF")
        weights = Text("Weights", font_size=20, color="#FFFF00")
        path = Dot(color="#FF00FF")
        grads = Rectangle(height=0.3, width=0.6, color="#00FF00")
        final = Text("Credit Assigned!", font_size=20, color="#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(network, "B2", "E5", scale_factor=0.6)
        self.play(FadeIn(network))
        self.place_in_area(derivs, "B2", "B5", scale_factor=0.7)
        self.play(FadeIn(derivs))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(weights, "C3", scale_factor=0.8)
        self.play(FadeIn(weights))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.place_in_area(path, "C2", "E2", scale_factor=0.6)
        self.play(FadeIn(path))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.place_in_area(grads, "D4", "E6", scale_factor=0.7)
        self.play(FadeIn(grads))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.place_at_grid(final, "F3", scale_factor=1.0)
        self.play(Write(final))
        
        self.wait(2)
