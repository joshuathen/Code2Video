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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Leap of Abstraction", [
            "Abstract vector spaces focus on behavior, not appearance.",
            "A vector is anything that follows specific algebraic rules.",
            "Think of these rules as membership for a club."
        ])

        # === Animation for Lecture Line 1 ===
        # Display a #FFFFFF arrow icon labeled 'IS' and a #FF00FF circle icon labeled 'BEHAVES'.
        # Shift focus to 'BEHAVES'.
        
        self.lecture[0].set_color(WHITE)
        
        arrow_icon = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        self.place_at_grid(arrow_icon, "B2", scale_factor=0.6)
        label_is = Text("IS", font_size=20, color=WHITE)
        self.place_at_grid(label_is, "A2")
        
        circle_icon = Circle(color="#FF00FF", fill_opacity=0.8)
        self.place_at_grid(circle_icon, "B5", scale_factor=0.4)
        label_behaves = Text("BEHAVES", font_size=20, color="#FF00FF")
        self.place_at_grid(label_behaves, "A5")
        
        self.play(FadeIn(arrow_icon), FadeIn(label_is), FadeIn(circle_icon), FadeIn(label_behaves))
        self.wait(1)
        
        # Shift focus to 'BEHAVES'
        self.play(
            arrow_icon.animate.set_opacity(0.2),
            label_is.animate.set_opacity(0.2),
            circle_icon.animate.scale(1.2),
            label_behaves.animate.scale(1.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the #FF00FF circle into a 'Digital Chameleon' that changes color to #00FFFF. 
        # Label it 'Color Vector'.
        
        self.lecture[1].set_color("#00FFFF")
        
        # Creating a stylized Chameleon shape: a RoundedRectangle with small circles for eyes
        chameleon_body = RoundedRectangle(corner_radius=0.2, height=0.6, width=0.8, color="#FF00FF", fill_opacity=1)
        eye1 = Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(chameleon_body.get_center() + [0.2, 0.1, 0])
        eye2 = Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(chameleon_body.get_center() + [0.2, -0.1, 0])
        chameleon = VGroup(chameleon_body, eye1, eye2)
        self.place_at_grid(chameleon, "B5")

        label_color_vector = Text("Color Vector", font_size=20, color="#00FFFF")
        self.place_at_grid(label_color_vector, "A5")

        self.play(
            Transform(circle_icon, chameleon_body),
            FadeIn(eye1), FadeIn(eye2),
            FadeOut(label_behaves),
            FadeIn(label_color_vector),
            self.lecture[0].animate.set_color(GRAY)
        )
        self.wait(0.5)
        
        # Change color to #00FFFF
        self.play(
            circle_icon.animate.set_color("#00FFFF"),
            chameleon_body.animate.set_color("#00FFFF"),
            label_color_vector.animate.set_color("#00FFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the 'Color Vector' undergoing addition with another color and scaling in intensity, 
        # both resulting in valid colors.
        
        self.lecture[2].set_color(YELLOW)
        
        # Addition: Cyan + Blue = Deep Cyan
        color1 = Square(side_length=0.4, color="#00FFFF", fill_opacity=1)
        plus = Text("+", font_size=20)
        color2 = Square(side_length=0.4, color="#0000FF", fill_opacity=1)
        equals1 = Text("=", font_size=20)
        result1 = Square(side_length=0.4, color="#0080FF", fill_opacity=1)
        
        addition_group = VGroup(color1, plus, color2, equals1, result1).arrange(RIGHT, buff=0.2)
        self.place_at_grid(addition_group, "D2", scale_factor=1.0)
        
        # Scaling: 2 * Red = Bright Red
        scalar = Text("2 \u00d7", font_size=20) 
        color3 = Square(side_length=0.4, color="#800000", fill_opacity=1)
        equals2 = Text("=", font_size=20)
        result2 = Square(side_length=0.4, color="#FF0000", fill_opacity=1)
        
        scaling_group = VGroup(scalar, color3, equals2, result2).arrange(RIGHT, buff=0.2)
        self.place_at_grid(scaling_group, "D5", scale_factor=1.0)
        
        self.play(
            FadeIn(addition_group),
            FadeIn(scaling_group),
            self.lecture[1].animate.set_color(GRAY)
        )
        self.wait(2)
