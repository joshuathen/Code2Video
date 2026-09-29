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
        self.setup_layout("Application: The Poincaré Conjecture and Beyond", [
            "Topology helps in 3D modeling.", 
            "Character rigging requires manifold surfaces.", 
            "Movement must not rip surfaces. [Asset: 3d_model_rig]"
        ])
        
        # Define objects
        # 3D sphere using ThreeDScene primitives or basic Sphere (ThreeDScene not inherited but sphere is supported)
        sphere = Sphere(radius=1.5, fill_opacity=0.3, color=BLUE)
        character = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png")
        
        # Loop on sphere
        phi = PI / 2
        loop = ParametricFunction(
            lambda t: np.array([
                1.5 * np.sin(phi) * np.cos(t),
                1.5 * np.sin(phi) * np.sin(t),
                1.5 * np.cos(phi)
            ]),
            t_range=[0, 2 * PI],
            color=YELLOW
        )
        
        # Place objects
        self.place_at_grid(sphere, 'B5', scale_factor=0.7)
        self.place_at_grid(character, 'B5', scale_factor=0.1)
        self.place_at_grid(loop, 'D5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(sphere), FadeIn(character))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(Create(loop))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(loop.animate.scale(0.01).move_to(character.get_center()))
        self.play(FadeOut(loop))
