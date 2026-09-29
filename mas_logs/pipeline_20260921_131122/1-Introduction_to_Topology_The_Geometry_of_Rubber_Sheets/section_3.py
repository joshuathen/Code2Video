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
        self.setup_layout("The Key Invariant: Genus (The 'Hole' Count)", 
                          ["Genus is the hole count.", 
                           "It is a topological invariant.", 
                           "Different genus means distinct topology.", 
                           "Soccer ball has zero holes.", 
                           "Donut has one hole."])
        
        # Objects using SVGs as per assets
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/soccerball.svg")
        donut = Torus(major_radius=0.5, minor_radius=0.2, color=GREEN).set_opacity(0.6)
        double_torus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/donut.svg")

        label0 = Text("Genus 0", font_size=20, color=WHITE)
        label1 = Text("Genus 1", font_size=20, color=GREEN)
        label2 = Text("Genus 2", font_size=20, color="#FF4500")

        # === Animation for Lecture Line 1: Genus is the hole count. ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(sphere, 'B3', scale_factor=0.8)
        self.place_at_grid(label0, 'A3', scale_factor=0.8)
        self.play(Create(sphere), Write(label0))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2: It is a topological invariant. ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(donut, 'B4', scale_factor=0.8)
        self.place_at_grid(label1, 'A4', scale_factor=0.8)
        self.play(Create(donut), Write(label1))
        self.play(self.lecture[1].animate.set_color(GREEN))

        # === Animation for Lecture Line 3: Different genus means distinct topology. ===
        self.play(FadeIn(self.lecture[2]))
        self.place_at_grid(double_torus, 'B5', scale_factor=0.8)
        self.place_at_grid(label2, 'A5', scale_factor=0.8)
        self.play(Create(double_torus), Write(label2))
        self.play(self.lecture[2].animate.set_color("#FF4500"))

        # === Animation for Lecture Line 4: Soccer ball has zero holes. ===
        self.play(FadeIn(self.lecture[3]))
        self.play(sphere.animate.shift(UP * 0.1).set_color(GOLD), run_time=1)
        self.play(sphere.animate.shift(DOWN * 0.1).set_color(WHITE), run_time=1)
        self.play(self.lecture[3].animate.set_color(GOLD))

        # === Animation for Lecture Line 5: Donut has one hole. ===
        self.play(FadeIn(self.lecture[4]))
        self.play(donut.animate.shift(UP * 0.1).set_color(GOLD), run_time=1)
        self.play(donut.animate.shift(DOWN * 0.1).set_color(GREEN), run_time=1)
        self.play(self.lecture[4].animate.set_color(GOLD))
        self.wait(2)
