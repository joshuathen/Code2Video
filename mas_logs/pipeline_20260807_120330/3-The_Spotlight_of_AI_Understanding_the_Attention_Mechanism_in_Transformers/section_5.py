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
        # Data from storyboard
        title_text = "The Math: From Dot Product to Softmax"
        lecture_lines = [
            "Dot products calculate the similarity between Queries and Keys.",
            "These raw scores indicate how well words match.",
            "Softmax turns scores into percentages that sum to one.",
            "This creates a clear probability map of focus.",
            "The model pulls information based on these weights."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Formula 'Q · K = Similarity' appears on screen. Color L1 to #FFFF00.
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        formula = MathTex("Q", "\\cdot", "K", "=", "\\text{Similarity}", font_size=36)
        # Issue 36: Fixed scale_factor from 1.2 to 1.0
        self.place_at_grid(formula, "B3", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # An arrow points from the formula to a result score. Color L2 to #FFFF00.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        score_val = Text("8.5", font_size=36, color=YELLOW)
        self.place_at_grid(score_val, "C3", scale_factor=1.0)
        
        # Arrow from formula to score
        arrow = Arrow(start=formula.get_bottom(), end=score_val.get_top(), buff=0.1, color=WHITE)
        
        self.play(Create(arrow), FadeIn(score_val))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A 'Softmax' bar chart turns scores into percentages. Color L3 to #FFFF00.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Fade out previous visuals to clear the space
        self.play(FadeOut(formula), FadeOut(arrow), FadeOut(score_val))
        
        softmax_label = Text("Softmax", font_size=28, color="#FFFF00")
        # Issue 35: Repositioned softmax_label from A3 to D2
        self.place_at_grid(softmax_label, "D2", scale_factor=1.0)
        
        # Create bars representing raw scores
        # Note: Bars are positioned such that they grow from the bottom of Row F
        bar_a = Rectangle(width=0.6, height=1.5, color=BLUE, fill_opacity=0.6)
        bar_b = Rectangle(width=0.6, height=1.0, color=RED, fill_opacity=0.6)
        
        # Position bars
        self.place_at_grid(bar_a, "E2", scale_factor=1.0)
        self.place_at_grid(bar_b, "E3", scale_factor=1.0)
        # Align them to the same base line (Row F)
        bar_a.align_to(self.grid["F2"], DOWN)
        bar_b.align_to(self.grid["F3"], DOWN)
        
        label_a = Text("A", font_size=20).next_to(bar_a, DOWN)
        label_b = Text("B", font_size=20).next_to(bar_b, DOWN)
        
        # Initial raw values
        val_a = DecimalNumber(8.5, num_decimal_places=1, font_size=24).next_to(bar_a, UP)
        val_b = DecimalNumber(6.0, num_decimal_places=1, font_size=24).next_to(bar_b, UP)
        
        self.play(
            FadeIn(softmax_label), 
            Create(bar_a), Create(bar_b), 
            Write(label_a), Write(label_b), 
            FadeIn(val_a), FadeIn(val_b)
        )
        self.wait(0.5)
        
        # Animate the transformation into percentages (Softmax effect)
        # Bar heights change to reflect 0.92 and 0.08
        self.play(
            bar_a.animate.stretch_to_fit_height(3.0).align_to(self.grid["F2"], DOWN),
            bar_b.animate.stretch_to_fit_height(0.3).align_to(self.grid["F3"], DOWN),
            val_a.animate.set_value(0.92),
            val_b.animate.set_value(0.08),
            run_time=1.5
        )
        # Readjust value labels to sit atop the new bar heights
        self.play(
            val_a.animate.next_to(bar_a, UP),
            val_b.animate.next_to(bar_b, UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A large data block flows from high-percentage 'Value'. Color L4 to #FFFF00.
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Represent the retrieval from a "Value" block
        value_box = Square(side_length=0.8, color=GREEN, fill_opacity=0.7)
        value_text = Text("Value A", font_size=18).move_to(value_box)
        self.place_at_grid(value_box, "C5", scale_factor=1.0)
        
        output_box = Square(side_length=0.8, color=PURPLE, fill_opacity=0.7)
        output_text = Text("Output", font_size=18).move_to(output_box)
        self.place_at_grid(output_box, "E5", scale_factor=1.0)
        
        # Small yellow block representing data flowing
        data_packet = Rectangle(width=0.4, height=0.2, color=YELLOW, fill_opacity=1.0).move_to(value_box.get_center())
        
        self.play(FadeIn(value_box), FadeIn(value_text), FadeIn(output_box), FadeIn(output_text))
        self.play(data_packet.animate.move_to(output_box.get_center()), run_time=1.2, rate_func=smooth)
        self.play(FadeOut(data_packet))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A probability map (#00FFFF) overlays the word sequence. Color L5 to #FFFF00.
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        
        # Clear specific chart/flow elements
        self.play(
            FadeOut(softmax_label), FadeOut(bar_a), FadeOut(bar_b), 
            FadeOut(label_a), FadeOut(label_b), FadeOut(val_a), FadeOut(val_b),
            FadeOut(value_box), FadeOut(value_text), FadeOut(output_box), FadeOut(output_text)
        )
        
        # Sequence of words for the final visualization
        sentence_words = ["The", "AI", "Spotlight", "is", "Bright"]
        word_mobs = VGroup(*[Text(w, font_size=28) for w in sentence_words]).arrange(RIGHT, buff=0.4)
        # Issue 34: Adjusted area and scale to prevent crowding
        self.place_in_area(word_mobs, "B2", "B6", scale_factor=0.9)
        
        # Visualizing the probability map using highlights of varying opacity
        highlights = VGroup()
        focus_weights = [0.1, 0.9, 0.8, 0.2, 0.1]
        for idx, word_mob in enumerate(word_mobs):
            highlight_rect = Rectangle(
                width=word_mob.get_width() + 0.2,
                height=word_mob.get_height() + 0.2,
                fill_color="#00FFFF",
                fill_opacity=focus_weights[idx],
                stroke_width=0
            ).move_to(word_mob)
            highlights.add(highlight_rect)
            
        self.play(Write(word_mobs))
        self.play(FadeIn(highlights))
        self.wait(2)
