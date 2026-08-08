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
        self.setup_layout("The Power of Abstraction", [
            "Abstraction provides a unified mathematical framework.",
            "Properties proven once apply to all spaces.",
            "Diverse systems share the same underlying logic."
        ])
        
        # Mobjects for animations
        # Assets
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg", color=BLUE)
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg", color=YELLOW)
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=PURPLE)
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg", color=ORANGE)
        
        abstract_shape = Circle(color=WHITE, fill_opacity=0.3)
        label = Text("Vector Space", color="#FF8080", font_size=24)
        
        # === Animation for Lecture Line 1 ===
        # Show multiple disparate objects transforming into the same abstract shape.
        self.place_at_grid(bicycle, 'B2', scale_factor=0.6)
        self.place_at_grid(airplane, 'B4', scale_factor=0.7)
        self.place_at_grid(camera, 'B6', scale_factor=0.7)
        self.play(Create(bicycle), Create(airplane), Create(camera))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(
            Transform(bicycle, abstract_shape.copy()),
            Transform(airplane, abstract_shape.copy()),
            Transform(camera, abstract_shape.copy())
        )

        # === Animation for Lecture Line 2 ===
        # Display the label 'Vector Space' hovering above them.
        self.place_at_grid(label, 'A5', scale_factor=0.9)
        self.play(FadeIn(label))
        self.play(self.lecture[1].animate.set_color("#FF8080"))

        # === Animation for Lecture Line 3 ===
        # Fade out individual objects, leaving only the abstract structure.
        # Note: bicycle, airplane, camera were transformed into copies of abstract_shape.
        self.play(FadeOut(bicycle), FadeOut(airplane), FadeOut(camera))
        self.place_in_area(abstract_shape, 'D3', 'E5', scale_factor=1.2)
        self.play(FadeIn(abstract_shape))
        self.play(self.lecture[2].animate.set_color("#80FF80"))
        self.wait(2)
