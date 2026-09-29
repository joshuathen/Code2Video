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
        lecture_lines = ["We synthesize topology and numerical methods.", "Detect roots with winding numbers.", "Visualize using color, pin with Newton."]
        self.setup_layout("Synthesis & Summary", lecture_lines)
        
        # Colors for highlights
        colors = ["#FF7F50", "#00CED1", "#DA70D6"]

        # === Animation for Lecture Line 1 ===
        # Represent topology and numerical methods
        top_mesh = VGroup(*[Circle(radius=0.1, color=BLUE, fill_opacity=0.5) for _ in range(10)])
        top_mesh.arrange_in_grid(2, 5)
        num_formula = MathTex(r"f(z) = 0", font_size=36)
        
        self.place_at_grid(top_mesh, "B2")
        # Issue 35 fix:
        self.place_at_grid(num_formula, "A5", scale_factor=0.9)
        
        self.play(self.lecture[0].animate.set_color(colors[0]), Create(top_mesh), Write(num_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        winding_curve = ParametricFunction(
            lambda t: np.array([0.3 * np.cos(t), 0.3 * np.sin(t), 0]), t_range=[0, 2*PI]
        )
        winding_curve.set_color(YELLOW)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pin.svg]
        root_indicator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pin.svg")
        
        self.place_at_grid(winding_curve, "E2")
        # Issue 33 fix:
        self.place_at_grid(root_indicator, "E3", scale_factor=0.7)
        
        self.play(self.lecture[1].animate.set_color(colors[1]), Create(winding_curve), FadeIn(root_indicator))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Issue 34 fix:
        color_map = Square(side_length=1.5, fill_opacity=0.8, color=PURPLE)
        newton_path = Arrow(start=UP*0.5, end=DOWN*0.5, color=WHITE)
        
        self.place_in_area(color_map, "E5", "F6", scale_factor=0.6)
        self.place_at_grid(newton_path, "D5")
        
        self.play(self.lecture[2].animate.set_color(colors[2]), FadeIn(color_map), GrowArrow(newton_path))
        self.wait(2)
