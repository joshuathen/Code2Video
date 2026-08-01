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
        title = "The Wordle Puzzle: A Search for Information"
        lines = [
            'Wordle begins with five empty slots for a hidden word.',
            'We start with a cloud of 2,300 potential words.',
            'A detective searches this massive word list for clues.',
            'Each guess reveals a pattern of green and yellow tiles.',
            "This feedback eliminates words that don't fit the pattern."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Create five gray (#808080) squares in the center.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        squares = VGroup(*[Square(side_length=0.6, color="#808080", stroke_width=4) for _ in range(5)])
        squares.arrange(RIGHT, buff=0.2)
        
        # Fix (Issue 36): Position squares at B1-B6 to avoid overlap
        self.place_in_area(squares, 'B1', 'B6', scale_factor=0.9)
        
        self.play(Create(squares), run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Generate a cloud of 2,300 small white (#FFFFFF) text labels.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Representing 2300 words with dots and a sample of labels
        cloud_dots = VGroup()
        for _ in range(150):
            dot = Dot(radius=0.015, color="#FFFFFF", fill_opacity=0.4)
            x_pos = np.random.uniform(0.5, 5.5)
            y_pos = np.random.uniform(-2.8, 2.2)
            dot.move_to([x_pos, y_pos, 0])
            cloud_dots.add(dot)

        cloud_labels = VGroup()
        word_list = ["REBUT", "SCARE", "THOSE", "PLATE", "LIGHT", "CRANE", "AUDIO", "STARE"]
        for _ in range(15):
            lbl = Text(str(np.random.choice(word_list)), font_size=10, color="#FFFFFF", fill_opacity=0.3)
            x_pos = np.random.uniform(0.5, 5.5)
            y_pos = np.random.uniform(-2.8, 2.2)
            lbl.move_to([x_pos, y_pos, 0])
            cloud_labels.add(lbl)

        self.play(FadeIn(cloud_dots), FadeIn(cloud_labels), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Detective [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/detective.svg] and magnifying glass scan cloud.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Asset Integration (Issue 32)
        detective = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/detective.svg")
        detective.set_color(WHITE)

        # Magnifying glass: white circle and line
        mag_circle = Circle(radius=0.3, color="#FFFFFF", stroke_width=3)
        mag_handle = Line(
            mag_circle.point_at_angle(-45*DEGREES), 
            mag_circle.point_at_angle(-45*DEGREES) + [0.3, -0.3, 0], 
            color="#FFFFFF", stroke_width=3
        )
        magnifying_glass = VGroup(mag_circle, mag_handle)
        
        # Position glass at A6 (Issue 37) and attach detective
        self.place_at_grid(magnifying_glass, 'A6', scale_factor=1.0)
        detective.scale(0.5).next_to(magnifying_glass, LEFT, buff=0.1)
        det_group = VGroup(detective, magnifying_glass)

        self.play(FadeIn(det_group))
        
        # Scanning path
        for target_pos in ["A2", "D5", "F2"]:
            self.play(det_group.animate.move_to(self.grid[target_pos]), run_time=0.8, rate_func=smooth)
        
        self.play(FadeOut(det_group))

        # === Animation for Lecture Line 4 ===
        # Flip squares to colors: Green (#00FF00), Gray (#808080), Yellow (#FFFF00), Gray, Gray.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        target_colors = ["#00FF00", "#808080", "#FFFF00", "#808080", "#808080"]
        reveal_anims = []
        for i, (sq, col) in enumerate(zip(squares, target_colors)):
            reveal_anims.append(sq.animate.set_fill(col, opacity=1.0).set_color(col).rotate(PI, axis=RIGHT))

        self.play(*reveal_anims, run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Fade out words/dots, leaving a small cluster of bright white (#FFFFFF) near squares.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )

        cluster_center = squares.get_center()
        
        # Group and filter cloud elements for the cluster effect
        keep_dots = VGroup(*[d for d in cloud_dots if np.linalg.norm(d.get_center() - cluster_center) < 1.5][:10])
        fade_dots = VGroup(*[d for d in cloud_dots if d not in keep_dots])
        
        keep_lbls = VGroup(*[l for l in cloud_labels if np.linalg.norm(l.get_center() - cluster_center) < 1.5][:3])
        fade_lbls = VGroup(*[l for l in cloud_labels if l not in keep_lbls])

        self.play(
            FadeOut(fade_dots),
            FadeOut(fade_lbls),
            keep_dots.animate.set_color(WHITE).scale(2).set_opacity(1),
            keep_lbls.animate.set_color(WHITE).scale(1.2).set_opacity(1),
            run_time=2
        )
        
        self.play(Flash(keep_dots, color=WHITE, flash_radius=0.6))
        self.wait(3)
