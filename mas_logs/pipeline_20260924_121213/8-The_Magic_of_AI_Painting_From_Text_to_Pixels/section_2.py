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
        lecture_lines = [
            "CLIP maps text and images into shared space.",
            "Semantic similarity brings vectors close together.",
            "Text and images align in mathematical space."
        ]
        self.setup_layout("Bridging Text and Vision: CLIP", lecture_lines)
        
        # Define colors from storyboard
        text_color = "#00FFFF"
        image_color = "#FF00FF"
        align_color = "#FFFF00"
        math_color = "#FFFFFF"
        loss_color = "#FF0000"

        # Initialize objects (once)
        text_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        text_vec = Dot(color=text_color)
        text_label = Text("Text Vector", font_size=18, color=text_color)
        text_group = VGroup(text_icon, text_vec, text_label).arrange(DOWN)
        
        image_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        image_vec = Dot(color=image_color)
        image_label = Text("Image Vector", font_size=18, color=image_color)
        image_group = VGroup(image_icon, image_vec, image_label).arrange(DOWN)
        
        group_container = VGroup(text_group, image_group)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(text_color))
        self.place_at_grid(text_group, 'B3', scale_factor=0.6)
        self.place_at_grid(image_group, 'C3', scale_factor=0.6)
        self.play(Create(text_group), Create(image_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(image_color))
        
        # Move them closer together as per B039 and layout feedback
        target_pos = self.grid['D5']
        self.play(
            text_group.animate.move_to(target_pos + UP*0.2),
            image_group.animate.move_to(target_pos + DOWN*0.2)
        )
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(align_color))
        
        align_line = Line(text_vec.get_center(), image_vec.get_center(), color=align_color)
        self.play(Create(align_line))
        
        # Show similarity calculation (positioned centrally per B040 and feedback)
        sim_text = Text("Cosine Similarity", font_size=16, color=math_color)
        self.place_at_grid(sim_text, 'D4', scale_factor=0.7)
        self.play(Write(sim_text))
        
        # Loss function label (positioned centrally per B040 and feedback)
        loss_text = Text("Contrastive Loss", font_size=16, color=loss_color)
        self.place_at_grid(loss_text, 'E4', scale_factor=0.7)
        self.play(FadeIn(loss_text))

        self.wait(2)
