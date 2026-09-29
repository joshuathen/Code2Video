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
        self.setup_layout("The Hook: The Broken Pencil Illusion", [
            "Observe the pencil inside this glass of water.", 
            "It appears broken at the water's surface.", 
            "Light rays change direction between media."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg]
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg]
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg", color=ORANGE)
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg", color=BLUE)
        
        pencil_group = VGroup(pencil, glass)
        self.place_in_area(pencil_group, 'B4', 'D5', scale_factor=0.9)
        
        label_pencil = Text("Pencil", font_size=18, color=WHITE)
        label_interface = Text("Interface", font_size=18, color="#FF0000")
        self.place_at_grid(label_pencil, 'B3', scale_factor=0.5)
        self.place_at_grid(label_interface, 'D4', scale_factor=0.5)
        
        self.play(Create(pencil_group), Write(label_pencil), Write(label_interface))
        self.lecture[0].set_color(ORANGE)

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg]
        ray1 = Line(ORIGIN, UP*0.5+RIGHT*0.5, color=YELLOW)
        ray2 = Line(ORIGIN, UP*0.5+LEFT*0.5, color=YELLOW)
        rays = VGroup(ray1, ray2).move_to(self.grid['C5'])
        label_rays = Text("Light Rays", font_size=18, color=YELLOW)
        self.place_at_grid(label_rays, 'E3', scale_factor=0.5)
        
        self.play(Create(rays), Write(label_rays))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg]
        refraction_effect = CurvedArrow(ORIGIN, UP*0.5+RIGHT*0.5, color=GREEN)
        self.place_at_grid(refraction_effect, 'D5', scale_factor=0.5)
        label_refraction = Text("Refraction", font_size=18, color=GREEN)
        self.place_at_grid(label_refraction, 'E5', scale_factor=0.5)
        
        self.play(Create(refraction_effect), Write(label_refraction))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
