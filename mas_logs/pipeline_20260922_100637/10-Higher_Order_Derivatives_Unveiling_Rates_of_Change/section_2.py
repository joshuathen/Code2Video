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
        lecture_lines = [
            "The second derivative is the derivative of velocity.",
            "It represents acceleration in physical motion.",
            "Acceleration describes how velocity itself changes."
        ]
        self.setup_layout("Defining the Second Derivative", lecture_lines)
        
        # Create elements
        curve = FunctionGraph(lambda x: 0.5 * x**2, x_range=[-2, 2], color="#FFFFFF")
        slope_markers = VGroup(*[Dot(point=curve.point_from_proportion(p), color="#FFFF00") for p in np.linspace(0.1, 0.9, 5)])
        accel_label = MathTex(r"f''(x)", color="#FF0000")
        
        try:
            car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        except:
            car = Dot(color=BLUE)
        
        try:
            rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg")
        except:
            rocket = Dot(color=RED)

        # Position elements
        self.place_in_area(curve, 'A3', 'C6', scale_factor=0.6)
        self.place_at_grid(slope_markers, 'C3', scale_factor=0.8)
        self.place_at_grid(accel_label, 'D3', scale_factor=1.0)
        self.place_at_grid(car, 'B1', scale_factor=0.3)
        self.place_at_grid(rocket, 'E5', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(Create(curve), FadeIn(car))
        self.play(MoveAlongPath(car, curve))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(slope_markers))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Write(accel_label), FadeIn(rocket))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
