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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Lex needs to complete this squirrel's story.",
            "It evaluates several possible words to fill the blank.",
            "Lex calculates that 'acorn' has the highest probability.",
            "The most likely word completes the sequence perfectly.",
            "LLMs are powerful statistical engines for predicting text."
        ]
        self.setup_layout("Introduction: The Prediction Game", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Lex Robot
        lex_head = Square(side_length=0.8, color="#00FFFF", fill_opacity=0.3)
        lex_eyes = VGroup(Dot(radius=0.05, color="#00FFFF"), Dot(radius=0.05, color="#00FFFF")).arrange(RIGHT, buff=0.2).move_to(lex_head.get_center())
        lex_antenna = Line(lex_head.get_top(), lex_head.get_top() + UP*0.2, color="#00FFFF")
        lex = VGroup(lex_head, lex_eyes, lex_antenna)
        lex_name = Text("Lex", font_size=18, color="#00FFFF").next_to(lex, DOWN, buff=0.1)
        lex_robot = VGroup(lex, lex_name)
        # Issue 34 fix: Using B2 grid position to avoid overcrowding
        self.place_at_grid(lex_robot, "B2", scale_factor=0.8)
        
        # Squirrel Asset (Issue 31 Integration)
        squirrel_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg")
        self.place_at_grid(squirrel_icon, "B4", scale_factor=0.8)
        
        # Sentence Text
        sentence_text = Text("The squirrel climbed the tree to hide its ___", color="#FFFFFF", font_size=20)
        # Issue 36 fix: Using C2 to C6 area for better margins
        self.place_in_area(sentence_text, "C2", "C6")
        
        self.play(FadeIn(lex_robot), FadeIn(squirrel_icon), Write(sentence_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line, reset first
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        opt_acorn = Text("acorn", color="#00FF00", font_size=24)
        opt_bicycle = Text("bicycle", color="#FF0000", font_size=24)
        opt_cloud = Text("cloud", color="#FF0000", font_size=24)
        
        self.place_at_grid(opt_acorn, "E2")
        self.place_at_grid(opt_bicycle, "E4")
        self.place_at_grid(opt_cloud, "E6")
        
        self.play(FadeIn(opt_acorn), FadeIn(opt_bicycle), FadeIn(opt_cloud))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line, reset second
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Probability label
        prob_label = Text("98% probability", color="#00FF00", font_size=18)
        prob_label.next_to(opt_acorn, DOWN, buff=0.1)
        
        # Scale acorn, fade others
        self.play(
            opt_acorn.animate.scale(1.3),
            FadeOut(opt_bicycle),
            FadeOut(opt_cloud),
            FadeIn(prob_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight fourth line, reset third
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Move acorn into the blank (rough position)
        target_pos = sentence_text.get_right() + LEFT*0.6
        
        self.play(
            opt_acorn.animate.move_to(target_pos).scale(0.8),
            FadeOut(prob_label)
        )
        
        # Transition to completed sentence
        final_sentence = Text(
            "The squirrel climbed the tree to hide its acorn", 
            color=WHITE, 
            font_size=20,
            t2c={"acorn": "#00FF00"}
        )
        # Issue 36 fix: Using C2 to C6 area
        self.place_in_area(final_sentence, "C2", "C6")
        
        self.play(
            FadeOut(sentence_text),
            FadeOut(opt_acorn),
            FadeIn(final_sentence)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight fifth line, reset fourth
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Top Label
        top_label = Text("LLMs are statistical prediction engines", color="#FFFF00", font_size=24)
        # Issue 35 fix: Using A2 to A6 area to avoid lecture note crowding
        self.place_in_area(top_label, "A2", "A6")
        
        self.play(Write(top_label))
        self.wait(2)
