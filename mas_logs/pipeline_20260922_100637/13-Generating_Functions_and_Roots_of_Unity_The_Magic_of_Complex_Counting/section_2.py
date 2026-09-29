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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Filter: Roots of Unity", [
            "Euler’s formula links algebra and geometry.",
            "Roots of unity filter modulo sums.",
            "Symmetric points on the unit circle."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Euler’s formula: e^(iθ) = cosθ + i sinθ
        euler_formula = MathTex(r"e^{i\theta} = \cos\theta + i \sin\theta", color="#FFCC00")
        self.place_at_grid(euler_formula, "B4", scale_factor=0.9)
        self.play(Write(euler_formula))
        self.play(self.lecture[0].animate.set_color("#FFCC00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Unit circle visual using compass
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle, "D3", "E5", scale_factor=1.2)
        self.place_at_grid(compass, "D2", scale_factor=0.5)
        
        k = 6
        points = []
        labels = []
        for j in range(k):
            angle = 2 * PI * j / k
            point = Dot(point=circle.get_center() + 1.5 * np.array([np.cos(angle), np.sin(angle), 0]), color="#00FFCC")
            points.append(point)
            label = Tex(f"$\omega^{j}$", font_size=20, color="#00FFCC")
            label.next_to(point, normalize(point.get_center() - circle.get_center()), buff=0.1)
            labels.append(label)
            
        self.play(DrawBorderThenFill(circle), FadeIn(compass))
        self.play(*[FadeIn(p) for p in points], *[Write(l) for l in labels])
        self.play(self.lecture[1].animate.set_color("#00FFCC"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotation illustration with protractor
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        self.place_at_grid(protractor, "C6", scale_factor=0.5)
        
        rotation = VGroup(*points, *labels)
        self.play(FadeIn(protractor))
        self.play(Rotate(rotation, angle=PI/3, about_point=circle.get_center()))
        self.play(self.lecture[2].animate.set_color("#FF66CC"))
        self.wait(2)
