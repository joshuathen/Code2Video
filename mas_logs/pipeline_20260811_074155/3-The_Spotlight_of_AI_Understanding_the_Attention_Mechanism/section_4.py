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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Math of Focus: Dot Products and Softmax", [
            "- Dot products measure alignment between Queries and Keys.",
            "- Softmax converts these scores into percentage weights.",
            "- Higher weights signal stronger focus on specific words.",
            "- The sum of all weights equals 100 percent.",
            "- This math directs the model's spotlight precisely."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Q (Eating, #00FFFF)
        q_label = Text("Q: 'Eating'", font_size=28, color="#00FFFF")
        self.place_at_grid(q_label, "C3") # Issue 27 fix
        
        # K1 (Robot, #FF6347) and K2 (Pizza, #FF6347)
        k1_label = Text("K1: 'Robot'", font_size=24, color="#FF6347")
        k2_label = Text("K2: 'Pizza'", font_size=24, color="#FF6347")
        self.place_at_grid(k1_label, "B6") # Issue 28 fix
        self.place_at_grid(k2_label, "D6") # Issue 28 fix

        # Visualizing vectors/connections
        arrow1 = Arrow(start=self.grid["C3"], end=self.grid["B6"], buff=0.4, color=WHITE)
        arrow2 = Arrow(start=self.grid["C3"], end=self.grid["D6"], buff=0.4, color=WHITE)

        self.play(FadeIn(q_label), FadeIn(k1_label), FadeIn(k2_label))
        self.play(Create(arrow1), Create(arrow2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Show the Dot Product calculation Q · K1 and Q · K2
        dot1 = MathTex("Q \\cdot K_1 = 0.1", font_size=32, color=WHITE)
        dot2 = MathTex("Q \\cdot K_2 = 0.9", font_size=32, color=WHITE)
        self.place_at_grid(dot1, "B6") # Issue 29 fix (overlaps label, but will indicate score)
        self.place_at_grid(dot2, "D6") # Issue 29 fix

        # We fade out labels to show scores
        self.play(
            FadeOut(k1_label), FadeOut(k2_label),
            FadeIn(dot1), FadeIn(dot2)
        )
        self.wait(1)
        
        # Pizza result glowing brighter
        self.play(Indicate(dot2, color=YELLOW, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Transition to Softmax bar chart
        self.play(
            FadeOut(arrow1), FadeOut(arrow2), 
            FadeOut(dot1), FadeOut(dot2),
            q_label.animate.scale(0.7).move_to(self.grid["A2"])
        )

        # Softmax layer represented by a bar chart
        bar_robot = Rectangle(height=0.4, width=1.0, fill_opacity=0.8, fill_color="#FF6347", stroke_width=2)
        bar_pizza = Rectangle(height=3.6, width=1.0, fill_opacity=0.8, fill_color="#00FFFF", stroke_width=2)
        
        # Group and position
        chart_bars = VGroup(bar_robot, bar_pizza).arrange(RIGHT, buff=0.8, aligned_edge=DOWN)
        self.place_in_area(chart_bars, "C4", "F5") # Issue 28 fix
        
        # Labels for bars
        label_robot = Text("Robot", font_size=20).next_to(bar_robot, DOWN, buff=0.2)
        label_pizza = Text("Pizza", font_size=20).next_to(bar_pizza, DOWN, buff=0.2)

        # Higher weights signal stronger focus
        self.play(Create(chart_bars), FadeIn(label_robot), FadeIn(label_pizza))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Scale bars to show 90% and 10% focus weights
        weight_robot = Text("10%", font_size=24).next_to(bar_robot, UP, buff=0.1)
        weight_pizza = Text("90%", font_size=24).next_to(bar_pizza, UP, buff=0.1)
        
        sum_check = MathTex("10\\% + 90\\% = 100\\%", font_size=36, color=YELLOW)
        self.place_in_area(sum_check, "B4", "B5") # Issue 29 fix

        self.play(FadeIn(weight_robot), FadeIn(weight_pizza))
        self.play(Write(sum_check))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Highlight the resulting focus on 'Pizza' with a glowing border (#FFFF00)
        focus_border = SurroundingRectangle(bar_pizza, color="#FFFF00", buff=0.1)
        
        self.play(Create(focus_border))
        self.play(focus_border.animate.set_stroke(width=8))
        self.wait(2)
        
        # Final cleanup / color reset
        self.lecture[4].set_color(WHITE)
        self.wait(2)
