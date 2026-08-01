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

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout with final script lines
        lecture_lines = [
            'Simply add one to return to zero.',
            'Five fundamental constants are now perfectly united.',
            'A single ring binds this mathematical masterpiece.',
            "Witness the ultimate elegance of Euler's identity.",
            'Simple, profound, and absolutely beautiful.'
        ]
        self.setup_layout("Summary and Elegance", lecture_lines)

        # Visual elements creation
        # Re-center visuals: axes and circle at grid D3 (Issue 35, 36)
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=2.5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": GREY}
        )
        self.place_at_grid(axes, 'D3')
        
        circle = Circle(radius=1.25, color=BLUE_A)
        self.place_at_grid(circle, 'D3')
        
        # Euler identity part 1: e^(i*pi) = -1 (Issue 37: Position at E3, scale 0.9)
        # Using Text instead of MathTex to ensure reliability
        identity1 = Text("e^(i*pi)=-1", font_size=36)
        self.place_at_grid(identity1, 'E3', scale_factor=0.9)
        
        # Identity part 2: e^(i*pi) + 1 = 0
        identity2 = Text("e^(i*pi)+1=0", font_size=36)
        self.place_at_grid(identity2, 'E3', scale_factor=0.9)
        
        # Setup colors for constants in identity2
        # Indices: e:0, ^:1, (:2, i:3, *:4, p:5, i:6, ):7, +:8, 1:9, =:10, 0:11
        identity2[0].set_color(BLUE)      # e
        identity2[3].set_color(PINK)      # i
        identity2[5:7].set_color(GOLD)    # pi
        identity2[9].set_color(WHITE)     # 1
        identity2[11].set_color(GRAY)     # 0
        
        # Load asset: glowing ring (Issue 25)
        ring_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/ring.svg"
        ring = SVGMobject(ring_asset_path, color=YELLOW)
        self.place_at_grid(ring, 'E3', scale_factor=1.8)

        # === Animation for Lecture Line 1 ===
        # Simply add one to return to zero.
        # Transform the expression e^(i*pi) = -1 into e^(i*pi) + 1 = 0.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(axes), Create(circle))
        self.play(Write(identity1))
        self.wait(1)
        self.play(ReplacementTransform(identity1, identity2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Five fundamental constants are now perfectly united.
        # Cycle through colors: e (blue), i (pink), pi (gold), 1 (white), 0 (gray).
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # identity2 is already colored, we scale the characters to highlight the cycle
        self.play(
            identity2[0].animate.scale(1.5),   # e
            identity2[3].animate.scale(1.5),   # i
            identity2[5:7].animate.scale(1.5), # pi
            identity2[9].animate.scale(1.5),   # 1
            identity2[11].animate.scale(1.5),  # 0
            run_time=1.5,
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A single ring binds this mathematical masterpiece.
        # Draw glowing ring around the identity.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(DrawBorderThenFill(ring))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Witness the ultimate elegance of Euler's identity.
        # Scale up the entire equation until it fills the screen, then scale back down.
        self.play(self.lecture[3].animate.set_color(YELLOW))
        id_group = VGroup(identity2, ring)
        self.play(id_group.animate.scale(3))
        self.wait(1)
        self.play(id_group.animate.scale(1/3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Simple, profound, and absolutely beautiful.
        # Slowly fade out everything except the white text of the identity.
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # Calculate target position for final centering (using grid anchor)
        target_pos = self.grid['C3'] 
        
        self.play(
            FadeOut(axes),
            FadeOut(circle),
            FadeOut(self.title),
            FadeOut(self.lecture),
            FadeOut(ring),
            identity2.animate.set_color(WHITE).scale(1.8).move_to(target_pos),
            run_time=3
        )
        self.wait(3)
