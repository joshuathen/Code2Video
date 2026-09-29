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
        self.setup_layout("Fourier Transform: The Mathematical Prism", [
            "A smoothie is just combined ingredients.", 
            "Signals are just combined sine waves.", 
            "Decompose them to find the ingredients."
        ])
        
        # Define assets
        blender = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blender.svg", color=WHITE)
        blender_label = Text("Blender", font_size=20)
        blender_group = VGroup(blender, blender_label).arrange(DOWN)
        
        # Ingredients
        ing1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fruit.svg", color="#FF5733")
        ing2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vegetable.svg", color="#33FF57")
        ing3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ice.svg", color="#3357FF")
        
        # Initial placement of blender
        self.place_in_area(blender_group, 'B4', 'D6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(blender_group))
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reposition ingredient labels
        self.place_at_grid(ing1, 'B4', scale_factor=0.6)
        self.place_at_grid(ing2, 'C4', scale_factor=0.6)
        self.place_at_grid(ing3, 'D4', scale_factor=0.6)
        
        self.play(
            FadeIn(ing1),
            FadeIn(ing2),
            FadeIn(ing3)
        )
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            ing1.animate.move_to(blender.get_center()),
            ing2.animate.move_to(blender.get_center()),
            ing3.animate.move_to(blender.get_center())
        )
        self.play(FadeOut(ing1), FadeOut(ing2), FadeOut(ing3))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
