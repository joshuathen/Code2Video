from manim import *
import os

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
        self.setup_layout("CLIP: The Multimodal Semantic Compass", [
            "CLIP maps text and images together.",
            "Concepts exist in a shared vector space.",
            "Related items cluster closely in this space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display 'CLIP Model' in center, color #00FFFF.
        clip_model = Text("CLIP Model", color="#00FFFF")
        self.place_at_grid(clip_model, "C3", scale_factor=0.6)
        self.play(Write(clip_model))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Load assets
        text_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/text.svg", color="#FFFFFF")
        image_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/image.svg", color="#FFFFFF")
        
        text_label = Text("Text", color="#FFFFFF")
        image_label = Text("Image", color="#FFFFFF")
        
        # Position icons and labels
        self.place_at_grid(text_icon, "B3", scale_factor=0.4)
        self.place_at_grid(image_icon, "D3", scale_factor=0.4)
        self.place_at_grid(text_label, "B4", scale_factor=0.6)
        self.place_at_grid(image_label, "D4", scale_factor=0.6)
        
        arrow_text = Arrow(clip_model.get_right(), text_icon.get_left(), color="#FFFFFF", buff=0.1)
        arrow_image = Arrow(clip_model.get_right(), image_icon.get_left(), color="#FFFFFF", buff=0.1)
        
        self.play(
            FadeIn(text_icon), FadeIn(text_label),
            FadeIn(image_icon), FadeIn(image_label),
            Create(arrow_text),
            Create(arrow_image)
        )
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        embedding_label = Text("Shared Embedding Space", color="#FF9900")
        self.place_in_area(embedding_label, "C4", "C6", scale_factor=0.65)
        
        self.play(Write(embedding_label))
        self.play(Indicate(embedding_label))
        self.lecture[2].set_color("#FF9900")
        
        self.wait(2)
