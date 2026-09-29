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
        lecture_lines = [
            "Diffusion models learn the art structure.",
            "CLIP learns the meaning of language.",
            "Together they turn text into pixels."
        ]
        self.setup_layout("Summary & Future Outlook", lecture_lines)
        
        # Colors for categories
        color_diffusion = "#FF9999" # Light red
        color_clip = "#99FF99"      # Light green
        color_final = "#9999FF"     # Light blue

        # === Animation for Lecture Line 1 ===
        # List key model milestones
        pixel_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg").set_color(WHITE)
        milestones = VGroup(
            Text("Latent Diffusion", font_size=24, color=color_diffusion),
            Text("Stable Diffusion", font_size=24, color=color_diffusion),
            Text("ControlNet", font_size=24, color=color_diffusion)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        group1 = VGroup(milestones, pixel_icon).arrange(RIGHT)
        self.place_at_grid(group1, 'B3', scale_factor=0.6)
        
        self.play(FadeIn(group1))
        self.play(self.lecture[0].animate.set_color(color_diffusion))

        # === Animation for Lecture Line 2 ===
        # Highlight future research directions
        future_research = VGroup(
            Text("Video Synthesis", font_size=24, color=color_clip),
            Text("Efficiency", font_size=24, color=color_clip),
            Text("Semantic Consistency", font_size=24, color=color_clip)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        self.place_at_grid(future_research, 'B5', scale_factor=0.6)
        self.play(FadeIn(future_research))
        self.play(self.lecture[1].animate.set_color(color_clip))

        # === Animation for Lecture Line 3 ===
        # Final system summary diagram
        monitor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg").set_color(WHITE)
        circle1 = Circle(radius=0.5, color=color_diffusion).set_fill(opacity=0.3)
        label1 = Text("Diffusion", font_size=20)
        system1 = VGroup(circle1, label1)
        
        circle2 = Circle(radius=0.5, color=color_clip).set_fill(opacity=0.3)
        label2 = Text("CLIP", font_size=20)
        system2 = VGroup(circle2, label2)
        
        system_nodes = VGroup(system1, system2).arrange(RIGHT, buff=1)
        summary_group = VGroup(system_nodes, monitor_icon).arrange(DOWN)
        
        self.place_in_area(summary_group, 'C4', 'D5', scale_factor=0.7)
        # Final positioning adjustment as suggested by issue 40/35
        self.place_at_grid(summary_group, 'E2', scale_factor=0.6)
        
        self.play(DrawBorderThenFill(system1), Write(label1))
        self.play(DrawBorderThenFill(system2), Write(label2))
        self.play(FadeIn(monitor_icon))
        self.play(self.lecture[2].animate.set_color(color_final))
        self.wait(2)
