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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup Section
        lecture_lines = [
            'Words become points on a massive multi-dimensional map.',
            'Similar meanings sit physically close together in space.',
            'We calculate relationships using precise mathematical vector distance.',
            '"King" minus "Man" plus "Woman" equals "Queen".',
            'This map transforms abstract IDs into semantic relationships.'
        ]
        self.setup_layout("Embeddings: The Semantic Map", lecture_lines)
        
        # Define word labels and dots
        # Colors: Puppy/Dog: #FFCC00, Toaster: #AAAAAA, Man/Woman Vector: #00FF00, King/Queen Vector: #FFD700
        dog_dot = Dot(color=WHITE, radius=0.08)
        
        # Asset: Puppy [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/puppy.svg]
        puppy_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/puppy.svg")
        puppy_svg.set_color(WHITE)
        
        toaster_dot = Dot(color=WHITE, radius=0.08)
        man_dot = Dot(color=WHITE, radius=0.08)
        woman_dot = Dot(color=WHITE, radius=0.08)
        king_dot = Dot(color=WHITE, radius=0.08)
        queen_dot = Dot(color=WHITE, radius=0.08)
        
        # Background dots for "Massive map"
        bg_dots = VGroup(*[Dot(color=GREY_E, radius=0.04) for _ in range(12)])
        bg_positions = ['A1', 'A4', 'A6', 'B4', 'B5', 'C1', 'C5', 'C6', 'D5', 'E4', 'F1', 'F4']
        for i, pos in enumerate(bg_positions):
            self.place_at_grid(bg_dots[i], pos)

        # Text labels
        dog_label = Text("Dog", font_size=20, color="#FFCC00")
        puppy_label = Text("Puppy", font_size=20, color="#FFCC00")
        toaster_label = Text("Toaster", font_size=20, color="#AAAAAA")
        man_label = Text("Man", font_size=20, color=WHITE)
        woman_label = Text("Woman", font_size=20, color=WHITE)
        king_label = Text("King", font_size=20, color=WHITE)
        queen_label = Text("Queen", font_size=20, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Grid positions
        self.place_at_grid(dog_dot, 'B2')
        # Issue 47/61: Puppy asset starts at B3 (close to Dog at B2)
        self.place_at_grid(puppy_svg, 'B3', scale_factor=0.3)
        self.place_at_grid(toaster_dot, 'B1')
        self.place_at_grid(man_dot, 'D2')
        self.place_at_grid(woman_dot, 'D3')
        self.place_at_grid(king_dot, 'E2')
        self.place_at_grid(queen_dot, 'E3')
        
        self.play(
            FadeIn(bg_dots),
            FadeIn(dog_dot), FadeIn(puppy_svg), FadeIn(toaster_dot),
            FadeIn(man_dot), FadeIn(woman_dot), FadeIn(king_dot), FadeIn(queen_dot),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFCC00")
        # Issue 48/61: puppy_label at A3 (aligned with B3)
        self.place_at_grid(dog_label, 'A2', scale_factor=0.8)
        self.place_at_grid(puppy_label, 'A3', scale_factor=0.8)
        
        self.play(
            dog_dot.animate.set_color("#FFCC00"),
            puppy_svg.animate.set_color("#FFCC00"),
            Write(dog_label),
            Write(puppy_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#AAAAAA")
        # Issue 46/61: toaster_label at A1 (near toaster_dot at B1)
        self.place_at_grid(toaster_label, 'A1', scale_factor=0.8)
        
        # Distant corner: Move dot to F6 and label to E6
        self.play(
            toaster_dot.animate.set_color("#AAAAAA").move_to(self.grid['F6']),
            Write(toaster_label),
            run_time=1.5
        )
        self.play(
            toaster_label.animate.move_to(self.grid['E6']),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        # Position labels within 1 grid unit of dots
        self.place_at_grid(man_label, 'C2', scale_factor=0.8)
        self.place_at_grid(woman_label, 'C3', scale_factor=0.8)
        self.place_at_grid(king_label, 'F2', scale_factor=0.8)
        self.place_at_grid(queen_label, 'F3', scale_factor=0.8)
        
        # Vectors
        man_woman_arrow = Arrow(
            start=self.grid['D2'], 
            end=self.grid['D3'], 
            buff=0.1, 
            color="#00FF00", 
            stroke_width=4
        )
        
        king_queen_arrow = Arrow(
            start=self.grid['E2'], 
            end=self.grid['E3'], 
            buff=0.1, 
            color="#FFD700", 
            stroke_width=4
        )

        self.play(
            Write(man_label),
            Write(woman_label),
            GrowArrow(man_woman_arrow),
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            Write(king_label),
            Write(queen_label),
            ReplacementTransform(man_woman_arrow.copy(), king_queen_arrow),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        self.play(
            Indicate(king_queen_arrow, color="#FFD700"),
            run_time=2
        )
        self.wait(2)
