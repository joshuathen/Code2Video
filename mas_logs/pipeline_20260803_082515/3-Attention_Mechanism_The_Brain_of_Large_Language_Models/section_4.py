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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "The Step-by-Step Calculation",
            [
                "First, calculate similarity by multiplying Query and Key vectors.",
                "These raw scores indicate how words relate to each other.",
                "Softmax converts these scores into normalized probability weights.",
                "Multiply weights by Value vectors to aggregate the information.",
                "The final output highlights the most relevant context."
            ]
        )

        # Asset path
        cat_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"

        # === Animation for Lecture Line 1 ===
        # Show text 'The cat sat' with a 'Query' pointer over 'cat' [Asset: cat.png] in #FFFFFF.
        self.lecture[0].set_color(WHITE)
        
        word_the = Text("The", font_size=24, color=WHITE)
        word_sat = Text("sat", font_size=24, color=WHITE)
        
        # Integrate Asset for 'cat'
        cat_icon = ImageMobject(cat_asset).scale(0.4)
        word_cat_label = Text("cat", font_size=20, color=WHITE)
        cat_obj = Group(cat_icon, word_cat_label).arrange(DOWN, buff=0.1)
        
        self.place_at_grid(word_the, "B3")
        self.place_at_grid(cat_obj, "B4")
        self.place_at_grid(word_sat, "B5")
        
        query_arrow = Arrow(DOWN, UP, color=WHITE, buff=0.1).scale(0.5)
        query_label = Text("Query", font_size=18, color=WHITE)
        query_ptr = VGroup(query_arrow, query_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(query_ptr, "C4")
        
        self.play(
            FadeIn(word_the),
            FadeIn(cat_obj),
            FadeIn(word_sat),
            FadeIn(query_ptr)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display numerical scores (10, 70, 20) over 'The', 'cat' [Asset: cat.png], and 'sat' in #FFFF00.
        self.lecture[1].set_color(YELLOW)
        
        score_10 = Text("10", font_size=24, color=YELLOW)
        score_70 = Text("70", font_size=24, color=YELLOW)
        score_20 = Text("20", font_size=24, color=YELLOW)
        
        self.place_at_grid(score_10, "A3")
        self.place_at_grid(score_70, "A4")
        self.place_at_grid(score_20, "A5")
        
        self.play(Write(score_10), Write(score_70), Write(score_20))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transform scores into a colored heatmap where 'cat' [Asset: cat.png] glows in #FF4500.
        self.lecture[2].set_color("#FF4500")
        
        # Heatmap rectangles as visual anchors behind the words
        heat_the = Rectangle(width=0.8, height=0.5, fill_color="#FF4500", fill_opacity=0.1, stroke_width=0)
        heat_cat = Rectangle(width=1.0, height=1.0, fill_color="#FF4500", fill_opacity=0.7, stroke_width=0)
        heat_sat = Rectangle(width=0.8, height=0.5, fill_color="#FF4500", fill_opacity=0.2, stroke_width=0)
        
        self.place_at_grid(heat_the, "B3")
        self.place_at_grid(heat_cat, "B4")
        self.place_at_grid(heat_sat, "B5")
        
        # Softmax weights replacing raw scores
        weight_10 = Text("0.1", font_size=20, color="#FF4500")
        weight_70 = Text("0.7", font_size=20, color="#FF4500")
        weight_20 = Text("0.2", font_size=20, color="#FF4500")
        
        self.place_at_grid(weight_10, "A3")
        self.place_at_grid(weight_70, "A4")
        self.place_at_grid(weight_20, "A5")
        
        self.play(
            FadeIn(heat_the), FadeIn(heat_cat), FadeIn(heat_sat),
            ReplacementTransform(score_10, weight_10),
            ReplacementTransform(score_70, weight_70),
            ReplacementTransform(score_20, weight_20)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Multiply weights by Value vectors to aggregate the information.
        val_color = "#90EE90"
        self.lecture[3].set_color(val_color)
        
        v1_box = Square(side_length=0.6, color=val_color, fill_opacity=0.5)
        v2_box = Square(side_length=0.6, color=val_color, fill_opacity=0.5)
        v3_box = Square(side_length=0.6, color=val_color, fill_opacity=0.5)
        
        v1_label = Text("V", font_size=20, color=WHITE).move_to(v1_box)
        v2_label = Text("V", font_size=20, color=WHITE).move_to(v2_box)
        v3_label = Text("V", font_size=20, color=WHITE).move_to(v3_box)
        
        v1 = VGroup(v1_box, v1_label)
        v2 = VGroup(v2_box, v2_label)
        v3 = VGroup(v3_box, v3_label)
        
        self.place_at_grid(v1, "D3")
        self.place_at_grid(v2, "D4")
        self.place_at_grid(v3, "D5")
        
        self.play(FadeIn(v1), FadeIn(v2), FadeIn(v3))
        
        # Scale based on weights 0.1, 0.7, 0.2
        self.play(
            v1.animate.scale(0.3),
            v2.animate.scale(1.1),
            v3.animate.scale(0.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Merge the scaled boxes into a single 'Output' vector in #00FFFF.
        self.lecture[4].set_color("#00FFFF")
        
        output_vec = Rectangle(width=1.0, height=1.4, color="#00FFFF", fill_opacity=0.8)
        output_txt = Text("Output", font_size=20, color=WHITE).move_to(output_vec)
        output_grp = VGroup(output_vec, output_txt)
        self.place_at_grid(output_grp, "E4")
        
        target_pos = self.grid["E4"]
        
        self.play(
            v1.animate.move_to(target_pos).set_opacity(0),
            v2.animate.move_to(target_pos).set_opacity(0),
            v3.animate.move_to(target_pos).set_opacity(0),
            FadeIn(output_grp, shift=UP)
        )
        self.wait(2)
