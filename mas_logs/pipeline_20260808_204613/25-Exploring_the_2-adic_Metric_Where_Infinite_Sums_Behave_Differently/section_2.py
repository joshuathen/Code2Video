from manim import *
import numpy as np

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
            "P-adic metrics define new distance.", 
            "Divisibility by two measures size.", 
            "High powers mean smaller values.", 
            "Binary trees visualize this structure.", 
            "Eight is smaller than three."
        ]
        self.setup_layout("Introducing the 2-adic Metric", lecture_lines)
        
        # Assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        scaler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display 2-adic valuation formula |x|_2 = 2^{-v_2(x)}
        formula = MathTex(r"|x|_2 = 2^{-v_2(x)}", color="#00FFCC")
        self.place_in_area(formula, 'A3', 'B5', scale_factor=1.0)
        self.place_at_grid(calculator, 'A2', scale_factor=0.5)
        self.play(Write(formula), FadeIn(calculator))
        self.lecture[0].set_color("#00FFCC")

        # === Animation for Lecture Line 2 ===
        # Visualize powers of 2 (2, 4, 8) shrinking
        nums = VGroup(
            Text("2", color=WHITE),
            Text("4", color=WHITE),
            Text("8", color=WHITE)
        ).arrange(RIGHT, buff=0.5)
        self.place_at_grid(nums, 'C3', scale_factor=0.8)
        self.place_at_grid(scaler, 'C2', scale_factor=0.5)
        self.play(FadeIn(nums), FadeIn(scaler))
        
        # Show shrinking
        self.play(
            nums[0].animate.scale(0.8),
            nums[1].animate.scale(0.6),
            nums[2].animate.scale(0.4),
        )
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # High powers mean smaller values
        arrow = Arrow(start=UP, end=DOWN, color="#FF99FF")
        self.place_at_grid(arrow, 'D5', scale_factor=0.8)
        self.place_at_grid(ruler, 'D6', scale_factor=0.5)
        self.play(GrowArrow(arrow), FadeIn(ruler))
        self.lecture[2].set_color("#FF99FF")

        # === Animation for Lecture Line 4 ===
        # Binary trees visualize this structure
        self.lecture[3].set_color(BLUE)

        # === Animation for Lecture Line 5 ===
        # Eight is smaller than three
        comp = MathTex(r"|8|_2 < |3|_2", color=YELLOW)
        self.place_in_area(comp, 'D3', 'E5', scale_factor=1.0)
        self.play(FadeIn(comp))
        self.lecture[4].set_color(YELLOW)
        
        self.wait(2)
