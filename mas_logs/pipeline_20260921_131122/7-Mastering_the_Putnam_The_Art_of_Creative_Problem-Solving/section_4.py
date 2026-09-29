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
        self.setup_layout("Integration: Bridging Calculus and Combinatorics", [
            "Integrate disparate fields for stronger solutions.", 
            "Riemann Sums bound discrete counting problems.", 
            "Geometric integrals model complex combinatorial structures."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Show title: Calculus & Combinatorics
        title_art = Tex(r"Calculus \& Combinatorics", color="#FFD700")
        self.place_at_grid(title_art, "B3", scale_factor=1.2)
        self.play(FadeIn(title_art))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Display integral symbol morphing into icon
        integral = MathTex(r"\int", color="#00CED1")
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg", color="#00CED1")
        self.place_at_grid(integral, "D2", scale_factor=1.5)
        self.play(Write(integral))
        self.play(Transform(integral, calc_icon.scale(1.5)))
        self.lecture[1].set_color("#00CED1")

        # === Animation for Lecture Line 3 ===
        # Combine curves, points, and bridge icon
        axes = Axes(x_range=[0, 3], y_range=[0, 3], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2/3, color="#FF4500")
        dots = VGroup(*[Dot(axes.c2p(i, (i**2)/3), color="#FF4500") for i in [0.5, 1, 1.5, 2, 2.5]])
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color="#FF4500")
        combined = VGroup(axes, curve, dots, bridge_icon)
        self.place_in_area(combined, "B4", "F6", scale_factor=0.5)
        self.play(Create(combined))
        self.lecture[2].set_color("#FF4500")
        self.wait(1)
