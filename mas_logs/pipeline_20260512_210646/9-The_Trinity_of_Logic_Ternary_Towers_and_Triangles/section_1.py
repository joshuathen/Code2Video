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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Foundation: The Power of Three (Ternary)", 
            [
                "Binary uses two states, but ternary uses three.", 
                "Ternary digits represent states zero, one, and two.", 
                "Positional values are powers of three: one, three, nine.", 
                "Three-state switches generate twenty-seven unique configurations.", 
                "This base-3 system coordinates our entire logical map."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Visualizing the difference (Binary vs Ternary labels) on Row A
        binary_label = Text("Binary: {0, 1}", font_size=24, color="#AAAAAA")
        ternary_label = Text("Ternary: {0, 1, 2}", font_size=24, color="#FFFFFF")
        
        self.place_at_grid(binary_label, "A2", scale_factor=0.8)
        self.place_at_grid(ternary_label, "A4", scale_factor=0.8)
        
        self.play(FadeIn(binary_label), FadeIn(ternary_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Display three circles labeled 0, 1, and 2 on Row B
        circle_0 = Circle(radius=0.4, color=WHITE)
        circle_1 = Circle(radius=0.4, color=WHITE)
        circle_2 = Circle(radius=0.4, color=WHITE)
        
        label_0 = Text("0", font_size=30)
        label_1 = Text("1", font_size=30)
        label_2 = Text("2", font_size=30)
        
        group_0 = VGroup(circle_0, label_0)
        group_1 = VGroup(circle_1, label_1)
        group_2 = VGroup(circle_2, label_2)
        
        self.place_at_grid(group_0, "B2")
        self.place_at_grid(group_1, "B3")
        self.place_at_grid(group_2, "B4")

        # Add switches icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/switches.svg]
        switches_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/switches.svg")
        self.place_at_grid(switches_icon, "B5", scale_factor=0.6)
        
        self.play(Create(group_0), Create(group_1), Create(group_2), FadeIn(switches_icon))
        
        # Sequentially highlight
        self.play(circle_0.animate.set_color("#00FF00"), label_0.animate.set_color("#00FF00"), run_time=0.5)
        self.play(circle_1.animate.set_color("#FFFF00"), label_1.animate.set_color("#FFFF00"), run_time=0.5)
        self.play(circle_2.animate.set_color("#FF0000"), label_2.animate.set_color("#FF0000"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Show magnitude labels: 3^2, 3^1, and 3^0 in #00FFFF on Row C
        mag_2 = Text("3²", color="#00FFFF", font_size=36)
        mag_1 = Text("3¹", color="#00FFFF", font_size=36)
        mag_0 = Text("3⁰", color="#00FFFF", font_size=36)
        
        self.place_at_grid(mag_2, "C2")
        self.place_at_grid(mag_1, "C3")
        self.place_at_grid(mag_0, "C4")
        
        # Row D for magnitude values
        mag_vals = VGroup(
            Text("(9)", font_size=20, color="#00FFFF"),
            Text("(3)", font_size=20, color="#00FFFF"),
            Text("(1)", font_size=20, color="#00FFFF")
        )
        
        self.place_at_grid(mag_vals[0], "D2")
        self.place_at_grid(mag_vals[1], "D3")
        self.place_at_grid(mag_vals[2], "D4")
        
        self.play(FadeIn(mag_2), FadeIn(mag_1), FadeIn(mag_0))
        self.play(Write(mag_vals))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Place digits '2', '1', and '0' under labels on Row E
        digit_2 = Text("2", font_size=42, color="#FFFFFF")
        digit_1 = Text("1", font_size=42, color="#FFFFFF")
        digit_0 = Text("0", font_size=42, color="#FFFFFF")
        
        self.place_at_grid(digit_2, "E2")
        self.place_at_grid(digit_1, "E3")
        self.place_at_grid(digit_0, "E4")
        
        # Show the calculation on Row F
        calc_text = Text("2 × 9 + 1 × 3 + 0 × 1", font_size=28, color="#FFFFFF")
        self.place_in_area(calc_text, "F2", "F5")
        
        self.play(Write(digit_2), Write(digit_1), Write(digit_0))
        self.play(FadeIn(calc_text))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Transform the ternary digits into decimal '21' and flash it
        result_21 = Text("= 21", font_size=42, color=WHITE)
        self.place_at_grid(result_21, "E5")
        
        final_group = VGroup(digit_2, digit_1, digit_0, calc_text, result_21)
        
        decimal_box = Text("Decimal: 21", font_size=48, color=WHITE)
        self.place_in_area(decimal_box, "D2", "F5")
        
        self.play(Write(result_21))
        self.wait(1)
        self.play(
            ReplacementTransform(final_group, decimal_box),
            FadeOut(group_0), FadeOut(group_1), FadeOut(group_2),
            FadeOut(switches_icon),
            FadeOut(mag_2), FadeOut(mag_1), FadeOut(mag_0), FadeOut(mag_vals),
            FadeOut(binary_label), FadeOut(ternary_label)
        )
        self.play(Indicate(decimal_box, color=WHITE, scale_factor=1.2))
        self.wait(2)
