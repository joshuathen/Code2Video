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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dimensions grow: points become lines, circles, then spheres.",
            "The distance formula tracks this growth across space.",
            "x-squared plus y-squared equals radius squared."
        ]
        self.setup_layout("Prerequisite Warm-up: Scaling Dimensions", lecture_lines)
        
        # Create objects
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FFFFFF")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#00CED1")
        r_label = Text("r", font_size=24, color="#FF69B4")
        
        # Group them for animation constraint
        anim_group = VGroup(circle, sphere, r_label)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(circle, 'B3', 'D4', scale_factor=0.9)
        self.play(Create(circle))
        self.lecture[0].set_color("#FFD700")
        
        # === Animation for Lecture Line 2 ===
        self.play(circle.animate.scale(1.5), run_time=1)
        self.lecture[1].set_color("#00CED1")
        
        # === Animation for Lecture Line 3 ===
        self.place_in_area(sphere, 'B3', 'D4', scale_factor=0.9)
        self.place_at_grid(r_label, 'B5', scale_factor=0.8)
        self.play(FadeIn(sphere), FadeIn(r_label))
        self.lecture[2].set_color("#FF69B4")
        
        # Apply layout requirement to entire animation group
        # Fix 21: self.place_in_area(group, 'C2', 'E5', scale_factor=0.85)
        # However, we already placed individual elements. We adjust the group position effectively.
        self.wait(1)
