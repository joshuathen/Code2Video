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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Linear Dependence: Redundancy", [
            "Linear dependence means having redundant vectors.", 
            "Redundant vectors don't add new dimensions.", 
            "Adding redundant vectors expands nothing."
        ])
        
        # Load assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        # === Animation for Lecture Line 1 ===
        # Display two vectors pointing in the same direction in #3357FF
        vec1 = Arrow(ORIGIN, RIGHT * 2, color="#3357FF")
        vec2 = Arrow(ORIGIN, RIGHT * 2.5, color="#3357FF")
        
        # Positioning requested by VideoCritic
        self.place_at_grid(vec1, 'B4', scale_factor=0.8)
        self.place_at_grid(vec2, 'E4', scale_factor=0.8)
        
        # Labels
        vec1_label = Text("v1", font_size=24, color="#3357FF")
        vec2_label = Text("v2", font_size=24, color="#3357FF")
        self.place_at_grid(vec1_label, 'B3', scale_factor=0.5)
        self.place_at_grid(vec2_label, 'E3', scale_factor=0.5)
        
        # Compass for orientation
        self.place_at_grid(compass, 'C5', scale_factor=0.3)
        
        self.play(Create(vec1), Create(vec2), Write(vec1_label), Write(vec2_label), FadeIn(compass))
        self.lecture[0].set_color("#3357FF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate one vector fading out
        self.play(FadeOut(vec2), FadeOut(vec2_label))
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the remaining vector, measuring alignment
        self.place_at_grid(ruler, 'D5', scale_factor=0.3)
        self.play(vec1.animate.set_color("#33FF57"), vec1_label.animate.set_color("#33FF57"), FadeIn(ruler))
        self.lecture[2].set_color("#33FF57")
        self.wait(2)
