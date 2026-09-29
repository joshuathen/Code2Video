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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Rotating coordinates reveals the square.", "Non-square sets violate continuity.", "The zero point is inevitable."]
        self.setup_layout("The Symmetry Argument", lecture_lines)
        
        # Elements
        shape1 = Square(side_length=1.5, color=WHITE)
        shape2 = Square(side_length=1.5, color=WHITE)
        self.place_at_grid(shape1, 'B2', scale_factor=0.5)
        self.place_at_grid(shape2, 'C2', scale_factor=0.5)
        
        # Load asset
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        target_icon.set_color(GREEN)
        self.place_at_grid(target_icon, 'D2', scale_factor=0.7)
        target_icon.set_opacity(0)
        
        # Organize for area layout
        combined_group = VGroup(shape1, shape2, target_icon)
        self.place_in_area(combined_group, 'B2', 'D4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(shape1), FadeIn(shape2))
        self.lecture[0].set_color("#00FFFF")
        self.play(Rotate(shape1, angle=PI/4), Rotate(shape2, angle=PI/4))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000")
        self.play(FadeIn(target_icon))
        self.play(Indicate(target_icon))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        self.play(Transform(shape1, shape2))
        self.wait(1)
