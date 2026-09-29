from manim import *

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
            "Analyze symmetry to unlock the problem.",
            "Identify invariants to track system state.",
            "Apply the extremal principle to conclude."
        ]
        self.setup_layout("Conclusion: Developing Your Toolkit", lecture_lines)
        
        # Load asset
        toolkit_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/toolkit.svg")
        
        # Create Key icons
        key_colors = [BLUE_B, GREEN_B, YELLOW_B]
        keys = VGroup(*[
            Rectangle(width=0.8, height=0.4, color=c, fill_opacity=0.5) 
            for c in key_colors
        ])
        
        # Group keys and icon for area placement
        toolkit_group = VGroup(toolkit_svg, keys)
        self.place_in_area(toolkit_group, 'B3', 'D5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_B))
        self.play(FadeIn(keys[0]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN_B))
        self.play(FadeIn(keys[1]))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW_B))
        self.play(FadeIn(keys[2]))
        
        # Final Highlight
        extra_text = Text("Practice Makes Perfect", font_size=32, color=YELLOW)
        self.place_in_area(extra_text, 'E3', 'F5', scale_factor=0.9)
        self.play(Write(extra_text))
        
        # Final transition with icon
        self.play(FadeIn(toolkit_svg))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
