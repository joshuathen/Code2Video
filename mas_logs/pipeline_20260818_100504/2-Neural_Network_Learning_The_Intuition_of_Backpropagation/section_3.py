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
            "Backpropagation assigns blame for the total error.",
            "The chain rule reveals weight contributions to errors.",
            "It is like tracing errors back through gears.",
            "Calculus identifies exactly how to adjust each weight.",
            "Small backward steps effectively reduce the total mistake."
        ]
        self.setup_layout("The Core Intuition: The Chain Rule", lecture_lines)
        
        # Assets
        gears1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg")
        gears2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg")
        
        # Elements
        func_text = MathTex("f(g(h(x)))", color=WHITE)
        self.place_in_area(func_text, 'A2', 'B5', scale_factor=1.2)
        
        chain_rule = MathTex(
            "\\frac{df}{dg} \\cdot \\frac{dg}{dh} \\cdot \\frac{dh}{dx}", 
            color=YELLOW
        )
        self.place_in_area(chain_rule, 'C2', 'C5', scale_factor=1.0)
        
        link_text = Text("Error = $\\sum$ Weight Contributions", color=GREEN, font_size=24)
        self.place_at_grid(link_text, 'E2', scale_factor=0.9)
        
        ripple = MathTex("\\Delta x \\rightarrow \\Delta f", color=PURPLE)
        self.place_at_grid(ripple, 'E4', scale_factor=0.9)
        
        core = MathTex("\\text{Chain Rule} = \\text{Backprop}", color=TEAL)
        self.place_at_grid(core, 'F3', scale_factor=0.9)
        
        self.place_at_grid(gears1, 'A5', scale_factor=0.4)
        self.place_at_grid(gears2, 'E5', scale_factor=0.4)

        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(func_text), FadeIn(gears1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(FadeIn(chain_rule))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(Create(link_text))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(FadeIn(ripple), FadeIn(gears2))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.play(GrowFromCenter(core))

        self.wait(2)
