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
        self.setup_layout("The Chain Rule: The Nested Gears", [
            "Nested functions act like rotating gears.", 
            "Turning one gear moves the next one.", 
            "Multiply gear ratios for total speed."
        ])
        
        # Gear asset
        gear_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg"
        
        # === Animation for Lecture Line 1 ===
        gear1 = SVGMobject(gear_path, color="#FFA500")
        gear2 = SVGMobject(gear_path, color="#FFA500")
        gears = VGroup(gear1, gear2).arrange(RIGHT, buff=0.1)
        self.place_in_area(gears, "B2", "C5", scale_factor=1.5)
        self.play(FadeIn(gears))
        self.lecture[0].set_color("#FFA500")

        # === Animation for Lecture Line 2 ===
        gear1.set_color("#FF4500")
        gear2.set_color("#FF4500")
        
        # Animate gear rotation
        self.play(
            Rotate(gear1, angle=PI, rate_func=linear),
            Rotate(gear2, angle=-PI, rate_func=linear),
            run_time=2
        )
        self.lecture[1].set_color("#FF4500")

        # === Animation for Lecture Line 3 ===
        label1 = MathTex("f'(g(x))").set_color("#FFFFFF").scale(0.8)
        label2 = MathTex("g'(x)").set_color("#FFFFFF").scale(0.8)
        self.place_at_grid(label1, "E2")
        self.place_at_grid(label2, "E5")
        
        product_label = MathTex("f'(g(x)) \\cdot g'(x)").set_color("#FFFFFF").scale(1.2)
        self.place_at_grid(product_label, "F3")
        
        self.play(Write(label1), Write(label2))
        self.play(Indicate(product_label))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(1)
