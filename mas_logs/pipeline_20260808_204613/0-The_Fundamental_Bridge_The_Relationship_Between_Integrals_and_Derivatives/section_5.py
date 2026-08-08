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
            "Derivatives measure the rate of change.",
            "Integrals measure the total accumulation.",
            "These inverse operations underpin all of modern science."
        ]
        self.setup_layout("Summary & Application Check", lecture_lines)
        
        # --- Animation for Lecture Line 1 & 2 (Combined) ---
        # Venn Diagram components
        circle_d = Circle(color=BLUE, radius=0.8)
        label_d = Text("Rate", font_size=20).move_to(circle_d.get_center())
        venn_d = VGroup(circle_d, label_d)
        
        circle_i = Circle(color=RED, radius=0.8)
        label_i = Text("Accum", font_size=20).move_to(circle_i.get_center())
        venn_i = VGroup(circle_i, label_i)
        
        # Load Assets
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Positioning
        self.place_in_area(venn_d, 'B1', 'B2', scale_factor=0.5)
        self.place_in_area(venn_i, 'B4', 'B5', scale_factor=0.5)
        self.place_at_grid(car, 'C3', scale_factor=0.3)
        
        self.play(FadeIn(venn_d), FadeIn(venn_i), FadeIn(car))
        self.lecture[0].set_color(BLUE)
        self.lecture[1].set_color(RED)
        
        # --- Animation for Lecture Line 3 ---
        formula = MathTex(r"\\int f'(x) dx = f(x) + C", color="#FFFF00")
        odometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/odometer.svg")
        
        self.place_at_grid(formula, 'E2', scale_factor=0.8)
        self.place_at_grid(odometer, 'E5', scale_factor=0.5)
        
        self.play(Write(formula), FadeIn(odometer))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
