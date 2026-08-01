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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup Section 5
        self.setup_layout(
            "Comparison and Verification", 
            [
                "Which animal group is the tallest?", 
                "Penguins have the most, elephants have the least.", 
                "Ten animals in total are all accounted for!"
            ]
        )
        
        # Define Assets
        penguin_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/penguin.svg"
        elephant_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/elephant.svg"
        
        # Colors
        BLOCK_COLOR_1 = BLUE_C      # For stack of 3 (Lions/Other)
        BLOCK_COLOR_2 = ORANGE      # For stack of 5 (Penguins)
        BLOCK_COLOR_3 = PINK        # For stack of 2 (Elephants)
        HIGHLIGHT_GREEN = "#00FF00"
        HIGHLIGHT_RED = "#FF0000"
        
        # Create Stacks
        # Stack 1 (Value: 3)
        stack1_icon = Circle(radius=0.4, color=BLOCK_COLOR_1, fill_opacity=0.8)
        self.place_at_grid(stack1_icon, "F2")
        stack1_blocks = VGroup(*[Square(side_length=0.7, color=BLOCK_COLOR_1, fill_opacity=0.4) for _ in range(3)])
        for i, grid_pos in enumerate(["E2", "D2", "C2"]):
            self.place_at_grid(stack1_blocks[i], grid_pos)
        
        # Stack 2 (Value: 5) - Penguins
        penguin_icon = SVGMobject(penguin_path)
        self.place_at_grid(penguin_icon, "F3", scale_factor=0.6)
        stack2_blocks = VGroup(*[Square(side_length=0.7, color=BLOCK_COLOR_2, fill_opacity=0.4) for _ in range(5)])
        for i, grid_pos in enumerate(["E3", "D3", "C3", "B3", "A3"]):
            self.place_at_grid(stack2_blocks[i], grid_pos)
        penguin_group = VGroup(penguin_icon, stack2_blocks)
        
        # Stack 3 (Value: 2) - Elephants
        elephant_icon = SVGMobject(elephant_path)
        self.place_at_grid(elephant_icon, "F4", scale_factor=0.6)
        stack3_blocks = VGroup(*[Square(side_length=0.7, color=BLOCK_COLOR_3, fill_opacity=0.4) for _ in range(2)])
        for i, grid_pos in enumerate(["E4", "D4"]):
            self.place_at_grid(stack3_blocks[i], grid_pos)
        elephant_group = VGroup(elephant_icon, stack3_blocks)
        
        # Initial display
        self.add(stack1_icon, stack1_blocks, penguin_icon, stack2_blocks, elephant_icon, stack3_blocks)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # "Which animal group is the tallest?"
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # White glow/border for penguin stack
        white_glow = SurroundingRectangle(penguin_group, color=WHITE, buff=0.1, stroke_width=4)
        self.play(
            penguin_group.animate.scale(1.1),
            Create(white_glow),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Penguins have the most, elephants have the least."
        # Color line 2: Green for Penguins, Red for Elephants
        self.play(
            self.lecture[1][0:20].animate.set_color(HIGHLIGHT_GREEN),
            self.lecture[1][21:].animate.set_color(HIGHLIGHT_RED)
        )
        
        green_glow = SurroundingRectangle(penguin_group, color=HIGHLIGHT_GREEN, buff=0.1, stroke_width=6)
        red_glow = SurroundingRectangle(elephant_group, color=HIGHLIGHT_RED, buff=0.1, stroke_width=6)
        
        self.play(
            ReplacementTransform(white_glow, green_glow),
            Create(red_glow),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Ten animals in total are all accounted for!"
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Floating digits
        digit3 = Text("3", font_size=36, color=WHITE)
        digit5 = Text("5", font_size=36, color=WHITE)
        digit2 = Text("2", font_size=36, color=WHITE)
        
        self.place_at_grid(digit3, "C2")
        self.place_at_grid(digit5, "A3")
        self.place_at_grid(digit2, "D4")
        
        # Target equation
        equation = Text("3 + 5 + 2 = 10", font_size=40, color=WHITE)
        self.place_in_area(equation, "B5", "D6", scale_factor=1.2)
        
        # Parts of equation for animation
        # "3 + 5 + 2 = 10"
        # Indices: 0:'3', 1:'+', 2:'5', 3:'+', 4:'2', 5:'=', 6:'1', 7:'0' (Approx depending on spacing)
        # Using a simpler method: just move digits to final positions
        
        self.play(Write(digit3), Write(digit5), Write(digit2))
        self.wait(0.5)
        
        # Clear glows to focus on equation
        self.play(FadeOut(green_glow), FadeOut(red_glow))
        
        # Form equation
        self.play(
            digit3.animate.move_to(equation[0].get_center()),
            digit5.animate.move_to(equation[2].get_center()),
            digit2.animate.move_to(equation[4].get_center()),
            FadeIn(equation[1]), # +
            FadeIn(equation[3]), # +
            FadeIn(equation[5]), # =
            FadeIn(equation[6:]), # 10
            run_time=2
        )
        self.add(equation)
        self.remove(digit3, digit5, digit2)
        
        self.play(Indicate(equation, color=YELLOW))
        self.wait(3)
