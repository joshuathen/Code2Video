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
        # Setup the scene with title and lecture lines
        lecture_lines = [
            'Meet Lex, our hyper-intelligent library robot.',
            'He memorized patterns from every book ever written.',
            'Lex predicts connections without actually understanding meaning.'
        ]
        self.setup_layout("Introduction: The Library Robot", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFCC")
        
        # Background Bookshelves: A grid of rectangles
        shelves = VGroup()
        for row in ["A", "B", "C", "D", "E", "F"]:
            for col in ["1", "2", "3", "4", "5", "6"]:
                shelf = Rectangle(width=0.8, height=0.4, fill_color="#222222", fill_opacity=1, stroke_width=1, stroke_color=GREY_E)
                self.place_at_grid(shelf, f"{row}{col}")
                shelves.add(shelf)
        
        # Robot 'Lex' - Integration of SVG asset
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        lex_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        lex_svg.set_color("#00FFCC")
        lex_label = Text("Lex", font_size=18, color="#00FFCC")
        lex_robot = VGroup(lex_svg, lex_label).arrange(DOWN, buff=0.1)
        
        # Position Lex in the area B3 to C4 (Fix overlap - Issue 41, 59)
        self.place_in_area(lex_robot, 'B3', 'C4', scale_factor=1.0)

        self.play(FadeIn(shelves))
        self.play(FadeIn(lex_robot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )

        # Indicate a specific bookshelf segment
        target_shelf_index = 7 # Corresponds to grid B2
        highlighted_shelf = shelves[target_shelf_index]
        
        # Lex moves 'hand' (a simple line) towards it
        hand = Line(lex_robot.get_left(), highlighted_shelf.get_right(), color=WHITE, stroke_width=4)
        
        self.play(
            highlighted_shelf.animate.set_fill("#FFFF00"),
            Create(hand)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFCC")
        )

        # Clear bookshelves and Lex's hand for the next visualization
        # Also move Lex slightly out of the way for the row E visualization
        self.play(
            FadeOut(shelves),
            FadeOut(hand),
            lex_robot.animate.shift(LEFT * 0.5).scale(0.8)
        )

        # Create two rectangular 'books' side-by-side
        book1 = Rectangle(width=0.6, height=0.9, fill_color=BLUE_E, fill_opacity=1, stroke_color=WHITE)
        book2 = Rectangle(width=0.6, height=0.9, fill_color=BLUE_E, fill_opacity=1, stroke_color=WHITE)
        
        # Position adjustments (Issue 42, 43, 59)
        self.place_at_grid(book1, 'E2', scale_factor=0.9)
        self.place_at_grid(book2, 'E5', scale_factor=0.9)
        
        book1_label = Text("Pattern A", font_size=14).next_to(book1, DOWN, buff=0.1)
        book2_label = Text("Pattern B", font_size=14).next_to(book2, DOWN, buff=0.1)

        # Connecting line showing a pattern
        connecting_line = Line(book1.get_right(), book2.get_left(), color="#00FFCC", stroke_width=6)
        sparkle = Star(n=5, color="#00FFCC", fill_opacity=1).scale(0.15).move_to(connecting_line.get_center())

        self.play(
            FadeIn(book1), FadeIn(book1_label),
            FadeIn(book2), FadeIn(book2_label)
        )
        self.play(
            Create(connecting_line),
            FadeIn(sparkle)
        )
        self.wait(2)
