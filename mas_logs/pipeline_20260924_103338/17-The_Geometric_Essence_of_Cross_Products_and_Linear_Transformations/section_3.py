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
        self.setup_layout("Geometric Intuition: Orientation and Chirality", [
            "Orientation depends on the handedness of basis vectors.",
            "Positive results follow a counter-clockwise path.",
            "Negative results indicate a clockwise orientation."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Use SVG asset as required by storyboard and B018
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        hand.set_color("#FFFFFF")
        # Fixed per issue 27: move to C2
        self.place_at_grid(hand, "C2", scale_factor=0.8)
        
        # Explicit label per B018
        hand_label = Text("Right-Hand Tool", font_size=18, color="#FFFFFF")
        hand_label.next_to(hand, DOWN, buff=0.2)
        
        self.play(FadeIn(hand), FadeIn(hand_label))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Rotate the vectors
        v1 = Vector([1, 0, 0], color="#00FFFF")
        v2 = Vector([0, 1, 0], color="#00FFFF")
        v_group = VGroup(v1, v2)
        # Fixed per issue 28: move to D5
        self.place_at_grid(v_group, "D5", scale_factor=0.9)
        self.play(Create(v_group))
        self.play(Rotate(v_group, angle=PI/4, about_point=self.grid["D5"]))
        self.lecture[1].set_color("#00FFFF")
        
        # === Animation for Lecture Line 3 ===
        # Use SVG asset for negative orientation
        flip_indicator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        flip_indicator.set_color("#FF0000")
        # Fixed per issue 29: move to F4
        self.place_at_grid(flip_indicator, "F4", scale_factor=0.7)
        
        # Explicit label per B018
        flip_label = Text("Flipped Orientation", font_size=18, color="#FF0000")
        flip_label.next_to(flip_indicator, UP, buff=0.2)
        
        self.play(FadeIn(flip_indicator), FadeIn(flip_label))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
