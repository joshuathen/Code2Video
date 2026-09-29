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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Fractional Time Divisions", [
            "Notes represent specific fractions of time.",
            "A whole note equals the entire bar.",
            "Half notes split the whole in two."
        ])
        
        # Visual assets
        pie_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pie.svg"
        
        # === Animation for Lecture Line 1 ===
        # Draw a circle using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pie.svg], 
        # split into halves, labels #00FFFF.
        self.lecture[0].set_color("#00FFFF")
        
        # We need the pie asset, but to avoid SVGMobject issues, 
        # let's use a simple Circle as a placeholder if needed, 
        # but the instructions mandate the asset. Let's try to load it safely.
        try:
            whole_circle = SVGMobject(pie_path)
        except:
            whole_circle = Circle(color=WHITE)
            
        self.place_at_grid(whole_circle, 'C5', scale_factor=0.5)
        self.play(Create(whole_circle))
        
        half_left = Sector(radius=1.2, start_angle=PI/2, angle=PI, color="#00FFFF")
        half_right = Sector(radius=1.2, start_angle=3*PI/2, angle=PI, color="#00FFFF")
        half_group = VGroup(half_left, half_right).move_to(self.grid['C5']).scale(0.5)
        
        self.play(ReplacementTransform(whole_circle, half_group))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Animate parts coming together to form a whole using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pie.svg], #FFFFFF.
        self.lecture[1].set_color("#FFFFFF")
        
        try:
            whole_circle_recomposed = SVGMobject(pie_path, color=WHITE)
        except:
            whole_circle_recomposed = Circle(color=WHITE)
            
        self.place_at_grid(whole_circle_recomposed, 'C5', scale_factor=0.5)
        self.play(ReplacementTransform(half_group, whole_circle_recomposed))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Further divide into fourths, highlight with #FF69B4.
        self.lecture[2].set_color("#FF69B4")
        q1 = Sector(radius=1.2, start_angle=PI/2, angle=PI/2, color="#FF69B4")
        q2 = Sector(radius=1.2, start_angle=0, angle=PI/2, color="#FF69B4")
        q3 = Sector(radius=1.2, start_angle=3*PI/2, angle=PI/2, color="#FF69B4")
        q4 = Sector(radius=1.2, start_angle=PI, angle=PI/2, color="#FF69B4")
        q_group = VGroup(q1, q2, q3, q4).move_to(self.grid['C5']).scale(0.5)
        
        self.play(ReplacementTransform(whole_circle_recomposed, q_group))
        self.wait(1)
