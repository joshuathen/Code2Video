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
        self.setup_layout("Applications and Wrap-up", [
            "PDEs are the fundamental language of nature.",
            "They govern fluids, electromagnetism, and quantum fields.",
            "PDEs in flight controller simulations.",
            "Real-time solvers handle complex environmental turbulence.",
            "PDEs unlock mastery over physical reality."
        ])
        
        # === Animation for Lecture Line 1 ===
        text_banner = Text("PDEs: Nature's Language", font_size=36, color=WHITE)
        self.place_in_area(text_banner, 'A4', 'B6')
        self.play(Write(text_banner))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Use assets: /scratch/pawsey1357/jthen/Code2Video/assets/icon/droplet.svg and atom.svg
        fluid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/droplet.svg", color="#00FFFF")
        em_icon = FunctionGraph(lambda x: np.sin(x*3), color="#00FFFF")
        quantum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg", color="#00FFFF")
        
        self.place_at_grid(fluid_icon, 'C2', scale_factor=0.6)
        self.place_at_grid(em_icon, 'C4', scale_factor=0.6)
        self.place_at_grid(quantum_icon, 'C6', scale_factor=0.6)
        
        self.play(FadeIn(fluid_icon), FadeIn(em_icon), FadeIn(quantum_icon))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        drone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg", color=WHITE)
        self.place_at_grid(drone_icon, 'E3', scale_factor=0.7)
        self.play(Create(drone_icon))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Turbulence vectors
        vectors = VGroup(*[Arrow(start=ORIGIN, end=UP*0.5, color="#FF4500") for _ in range(3)])
        vectors.arrange(RIGHT, buff=0.2)
        self.place_in_area(vectors, 'E4', 'E6', scale_factor=0.6)
        self.play(FadeIn(vectors))
        self.lecture[3].set_color("#FF4500")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(text_banner))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
