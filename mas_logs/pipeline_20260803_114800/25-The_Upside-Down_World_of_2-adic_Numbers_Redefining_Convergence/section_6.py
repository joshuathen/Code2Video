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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title_text = "Conclusion: Geometry depends on the Ruler"
        lecture_lines = [
            "Convergence depends entirely on your choice of metric.",
            "Different rulers reveal different mathematical truths.",
            "P-adic numbers power modern cryptography and string theory."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Label 'Ruler = Metric' in #FFD700 along with icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/r.svg] appears center.
        self.lecture[0].set_color("#FFD700")
        
        metric_label = Text("Ruler = Metric", color="#FFD700", font_size=36)
        r_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/r.svg")
        r_icon.set_color("#FFD700")
        r_icon.scale(0.5)
        
        metric_group = VGroup(r_icon, metric_label).arrange(RIGHT, buff=0.3)
        # Positioned in the upper-middle area (B2-B5) of the right-side grid
        self.place_in_area(metric_group, "B2", "B5", scale_factor=0.8)
        
        self.play(FadeIn(metric_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Icons for 'Cryptography' (#00FF00) and 'Physics' (#FF00FF) fade in.
        self.lecture[1].set_color("#00FF00")
        
        # Create a simple representation of a Cryptography icon (Lock)
        crypto_base = Square(side_length=0.5, color="#00FF00")
        crypto_shackle = Arc(radius=0.25, start_angle=0, angle=PI, color="#00FF00").shift(UP*0.25)
        crypto_icon_shape = VGroup(crypto_base, crypto_shackle)
        crypto_label = Text("Cryptography", font_size=18, color="#00FF00").next_to(crypto_icon_shape, DOWN, buff=0.1)
        crypto_icon = VGroup(crypto_icon_shape, crypto_label)
        self.place_at_grid(crypto_icon, "D2", scale_factor=0.9)
        
        # Create a simple representation of a Physics icon (Atom)
        physics_nucleus = Circle(radius=0.1, fill_opacity=1, color="#FF00FF")
        physics_orbit1 = Ellipse(width=0.7, height=0.2, color="#FF00FF").rotate(45*DEGREES)
        physics_orbit2 = Ellipse(width=0.7, height=0.2, color="#FF00FF").rotate(-45*DEGREES)
        physics_icon_shape = VGroup(physics_nucleus, physics_orbit1, physics_orbit2)
        physics_label = Text("Physics", font_size=18, color="#FF00FF").next_to(physics_icon_shape, DOWN, buff=0.1)
        physics_icon = VGroup(physics_icon_shape, physics_label)
        self.place_at_grid(physics_icon, "D5", scale_factor=0.9)
        
        self.play(FadeIn(crypto_icon), FadeIn(physics_icon))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The sum '1+2+4+8...' glows and transforms into '-1'.
        self.lecture[2].set_color("#FFFF00")
        
        sum_tex = MathTex("1+2+4+8+\\dots", color=WHITE)
        # Positioned in the middle-lower area (E2-E5)
        self.place_in_area(sum_tex, "E2", "E5", scale_factor=1.0)
        
        self.play(Write(sum_tex))
        self.play(Indicate(sum_tex, color="#FFFF00", scale_factor=1.1))
        
        # Transformation target: -1
        result_tex = MathTex("-1", color="#FFFF00")
        self.place_in_area(result_tex, "E2", "E5", scale_factor=1.2)
        
        self.play(Transform(sum_tex, result_tex))
        self.wait(3)
