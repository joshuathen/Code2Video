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
        self.setup_layout("Summary: Why Light Bends", [
            "Refractive index is material response.",
            "Frequency dictates oscillation coupling.",
            "Resonance drives bending magnitude."
        ])
        
        # Define colors for lecture lines
        c1 = "#FF9999" # Light red
        c2 = "#99FF99" # Light green
        c3 = "#9999FF" # Light blue

        # Load SVG Assets
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")

        # Create display elements
        box1 = RoundedRectangle(corner_radius=0.1, height=0.8, width=2.5, color=c1, fill_opacity=0.2)
        text1 = Text("Material Response", font_size=20, color=c1)
        
        box2 = RoundedRectangle(corner_radius=0.1, height=0.8, width=2.5, color=c2, fill_opacity=0.2)
        text2 = Text("Frequency Coupling", font_size=20, color=c2)
        
        box3 = RoundedRectangle(corner_radius=0.1, height=0.8, width=2.5, color=c3, fill_opacity=0.2)
        text3 = Text("Resonance Effect", font_size=20, color=c3)

        # Place elements in area for better distribution
        self.place_in_area(VGroup(box1, text1), 'B2', 'B5', scale_factor=0.9)
        self.place_in_area(VGroup(box2, text2), 'D2', 'D5', scale_factor=0.9)
        self.place_in_area(VGroup(box3, text3), 'F2', 'F5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        self.play(Create(box1), Write(text1), FadeIn(prism.scale(0.5).move_to(self.grid['B6'])))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        self.play(Create(box2), Write(text2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        self.play(Create(box3), Write(text3), FadeIn(glass.scale(0.5).move_to(self.grid['F6'])))

        self.wait(2)
